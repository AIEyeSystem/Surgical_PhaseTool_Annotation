"""Schema-versioned storage for unified temporal and spatial video annotations.

The project manifest is intentionally model-agnostic.  It stores temporal
annotations and references to frame-level IVAS labels, while dense masks and
decoded frames remain separate assets inside the case directory.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from IVAS import __version__
from IVAS.branding import LEGACY_VIDEO_PROJECT_NAMES, VIDEO_PROJECT_NAME
from IVAS.utils.persistence import atomic_write_bytes, project_write_lock
from IVAS.video.merge import merge_projects, ProjectConflictError


SCHEMA_VERSION = 3
APP_VERSION = __version__
VALID_TEMPORAL_KINDS = {"interval", "point", "rating"}
VALID_STATUSES = {"draft", "ready", "reviewed"}
VALID_TRACKING_STATUSES = {
    "queued",
    "running",
    "staged",
    "committed",
    "cancelled",
    "failed",
    "rolled_back",
}


class MediaRelinkRequired(ValueError):
    """Raised when a manifest cannot be safely bound to its recorded media."""

    def __init__(self, project_path: Path, recorded_path: Path, reason: str) -> None:
        self.project_path = Path(project_path)
        self.recorded_path = Path(recorded_path)
        self.reason = str(reason)
        super().__init__(
            "Video project requires explicit media relinking ({}): {}".format(
                self.reason, self.recorded_path
            )
        )


def discover_video_projects(output_root: Path) -> List[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    projects = {
        path.resolve()
        for name in (VIDEO_PROJECT_NAME, *LEGACY_VIDEO_PROJECT_NAMES)
        for path in root.rglob(name)
        if path.is_file()
    }
    return sorted(projects, key=lambda path: str(path).casefold())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_id(prefix: str) -> str:
    return "{}_{}".format(prefix, uuid.uuid4().hex)


def safe_filename(value: str, fallback: str = "untitled") -> str:
    result = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value)).strip(" ._")
    return result[:96] or fallback


def media_fingerprint(path: Path, block_size: int = 1_048_576) -> str:
    """Return a stable-enough identity without hashing an entire large video."""

    source = Path(path)
    digest = hashlib.sha256()
    try:
        stat = source.stat()
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        with source.open("rb") as stream:
            digest.update(stream.read(block_size))
            if stat.st_size > block_size:
                stream.seek(max(0, stat.st_size - block_size))
                digest.update(stream.read(block_size))
    except OSError:
        digest.update(str(source).encode("utf-8", errors="replace"))
    return digest.hexdigest()


def case_directory(output_root: Path, media_path: Path) -> Path:
    media = Path(media_path)
    return Path(output_root) / "{}_{}".format(
        safe_filename(media.stem), media_fingerprint(media)[:10]
    )


def format_timecode(milliseconds: Optional[int]) -> str:
    if milliseconds is None:
        return "—"
    total = max(0, int(milliseconds))
    hours, remainder = divmod(total, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(
        hours, minutes, seconds, millis
    )


def format_legacy_timecode(milliseconds: Optional[int]) -> str:
    total_seconds = max(0, int(milliseconds or 0)) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def parse_timecode(value: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("Timecode cannot be empty")
    if re.fullmatch(r"\d+", text):
        return int(text)
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("Invalid timecode: {}".format(value))
    try:
        if len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        else:
            hours, minutes, seconds = parts
        total = (int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Invalid timecode: {}".format(value)) from error
    if not math.isfinite(total) or total < 0:
        raise ValueError("Invalid timecode: {}".format(value))
    return int(round(total))


@dataclass
class TemporalTemplate:
    id: str
    label: str
    kind: str
    category: str
    group: str = "Default"
    colour: str = ""
    start_hint: str = ""
    end_hint: str = ""
    score_options: List[int] = field(default_factory=list)
    source: str = ""
    required: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TemporalTemplate":
        data = dict(value)
        data.setdefault("group", "Default")
        data.setdefault("colour", "")
        data.setdefault("start_hint", "")
        data.setdefault("end_hint", "")
        data.setdefault("score_options", [])
        data.setdefault("source", "")
        data.setdefault("required", False)
        data["score_options"] = [int(item) for item in data["score_options"]]
        data["required"] = bool(data["required"])
        return cls(**data)


@dataclass
class ValidationIssue:
    severity: str
    message: str
    annotation_id: str = ""


def default_temporal_templates() -> List[TemporalTemplate]:
    """Return small editable defaults for a new unified project."""

    return [
        TemporalTemplate(
            id="phase_default",
            label="Phase",
            kind="interval",
            category="phase",
            group="Workflow",
        ),
        TemporalTemplate(
            id="task_default",
            label="Task",
            kind="interval",
            category="task",
            group="Workflow",
        ),
        TemporalTemplate(
            id="keyframe_default",
            label="Key frame",
            kind="point",
            category="keyframe",
            group="Review",
        ),
        TemporalTemplate(
            id="quality_default",
            label="Quality",
            kind="rating",
            category="assessment",
            group="Review",
            score_options=[0, 1, 2, 3, 4, 5],
        ),
    ]


def parse_interval_template(
    path: Path, category: Optional[str] = None
) -> List[TemporalTemplate]:
    source = Path(path)
    category_name = str(category or source.stem).strip()
    result: List[TemporalTemplate] = []
    group = "Default"
    group_number = 1
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, start=1):
            text = raw.strip()
            if not text or text.startswith("//"):
                continue
            parts = [part.strip() for part in text.split("#")]
            label = parts[0]
            if label == "--":
                group_number += 1
                group = "Group {}".format(group_number)
                continue
            if not label:
                continue
            result.append(
                TemporalTemplate(
                    id="tpl_{}_{}_{}".format(
                        safe_filename(category_name).lower(), line_number, uuid.uuid4().hex[:8]
                    ),
                    label=label,
                    kind="interval",
                    category=category_name,
                    group=group,
                    start_hint=parts[1] if len(parts) > 1 else "",
                    end_hint=parts[2] if len(parts) > 2 else "",
                    source=str(source.resolve()),
                )
            )
    return result


def parse_rating_template(
    path: Path, category: Optional[str] = None
) -> List[TemporalTemplate]:
    source = Path(path)
    category_name = str(category or source.stem).strip()
    result: List[TemporalTemplate] = []
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, start=1):
            text = raw.strip()
            if not text or text.startswith("//"):
                continue
            parts = [part.strip() for part in text.split("#")]
            if not parts[0]:
                continue
            try:
                score_count = int(parts[1]) if len(parts) > 1 else 2
            except ValueError:
                score_count = 2
            kind = "point" if score_count == 0 else "rating"
            result.append(
                TemporalTemplate(
                    id="tpl_{}_{}_{}".format(
                        safe_filename(category_name).lower(), line_number, uuid.uuid4().hex[:8]
                    ),
                    label=parts[0],
                    kind=kind,
                    category=category_name,
                    group="Key frames" if kind == "point" else "Ratings",
                    score_options=list(range(max(0, score_count))),
                    source=str(source.resolve()),
                )
            )
    return result


def interval_coverage_ms(
    annotations: Iterable["TemporalAnnotation"], duration_ms: int = 0
) -> int:
    limit = max(0, int(duration_ms))
    spans: List[Tuple[int, int]] = []
    for annotation in annotations:
        if annotation.kind != "interval" or annotation.end_ms is None:
            continue
        start = max(0, int(annotation.start_ms))
        end = max(start, int(annotation.end_ms))
        if limit:
            start = min(start, limit)
            end = min(end, limit)
        spans.append((start, end))
    if not spans:
        return 0
    spans.sort()
    covered = 0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    return covered + current_end - current_start


@dataclass
class TemporalAnnotation:
    id: str
    kind: str
    category: str
    label: str
    start_ms: int
    end_ms: Optional[int] = None
    status: str = "draft"
    note: str = ""
    frame_index: Optional[int] = None
    score: Optional[int] = None
    asset_path: str = ""
    template_id: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TemporalAnnotation":
        data = dict(value)
        data.setdefault("end_ms", None)
        data.setdefault("status", "draft")
        data.setdefault("note", "")
        data.setdefault("frame_index", None)
        data.setdefault("score", None)
        data.setdefault("asset_path", "")
        data.setdefault("template_id", "")
        data.setdefault("created_at", utc_now())
        data.setdefault("updated_at", data["created_at"])
        return cls(**data)


class UnifiedVideoProjectStore:
    """Single recoverable manifest for a video annotation case."""

    def __init__(self, project_path: Path, project: Dict[str, Any]) -> None:
        self.path = Path(project_path)
        self.project = project
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []
        self.is_dirty = False
        self.recovered_from_backup = False
        self.recovery_source: Optional[Path] = None
        before_normalise = copy.deepcopy(self.project)
        self._normalise()
        self.is_dirty = self.project != before_normalise
        self._persisted_project = copy.deepcopy(before_normalise)

    @classmethod
    def create(
        cls,
        project_path: Path,
        media_path: Path,
        duration_ms: int = 0,
        fps: float = 0.0,
        frame_count: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> "UnifiedVideoProjectStore":
        media = Path(media_path)
        if not media.is_file():
            raise ValueError("Video file does not exist: {}".format(media))
        now = utc_now()
        project = {
            "schema_version": SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "project_id": make_id("case"),
            "created_at": now,
            "updated_at": now,
            "media": {
                "type": "video",
                "path": str(media.resolve()),
                "file_name": media.name,
                "fingerprint": media_fingerprint(media),
                "duration_ms": int(max(0, duration_ms)),
                "fps": float(max(0.0, fps)),
                "frame_count": int(max(0, frame_count)),
                "width": int(max(0, width)),
                "height": int(max(0, height)),
            },
            "temporal_annotations": [],
            "temporal_templates": [
                asdict(template) for template in default_temporal_templates()
            ],
            "instances": [],
            "spatial_frames": [],
            "tracking_runs": [],
            "frame_map": {},
            "media_history": [],
            "audit": [],
        }
        store = cls(project_path, project)
        store._persisted_project = None
        store._append_audit("project_created", "Created unified video project")
        store.is_dirty = True
        return store

    @classmethod
    def open_or_create(
        cls,
        output_root: Path,
        media_path: Path,
        **media_info: Any
    ) -> "UnifiedVideoProjectStore":
        directory = case_directory(Path(output_root), Path(media_path))
        project_path = directory / VIDEO_PROJECT_NAME
        existing_path = next(
            (
                candidate
                for candidate in (
                    project_path,
                    *(directory / name for name in LEGACY_VIDEO_PROJECT_NAMES),
                )
                if candidate.is_file()
            ),
            None,
        )
        if existing_path is not None:
            store = cls.load(existing_path)
            expected = media_fingerprint(Path(media_path))
            if store.media["fingerprint"] != expected:
                raise ValueError("Project media fingerprint does not match the video")
            store._update_media_path(Path(media_path))
            store.update_media_info(**media_info)
            return store
        legacy_path = directory / "project.surgui.json"
        if legacy_path.is_file():
            return cls.migrate_surgui_project(
                project_path, legacy_path, media_path, **media_info
            )
        return cls.create(project_path, media_path, **media_info)

    @classmethod
    def open_existing(
        cls,
        project_path: Path,
        media_path: Optional[Path] = None,
        allow_identity_change: bool = False,
    ) -> "UnifiedVideoProjectStore":
        """Open an arbitrary manifest and bind media only after identity checks."""

        store = cls.load(Path(project_path))
        recorded = Path(store.media["path"])
        candidate = Path(media_path).resolve() if media_path is not None else recorded
        if not candidate.is_file():
            raise MediaRelinkRequired(store.path, recorded, "recorded media is missing")
        fingerprint = media_fingerprint(candidate)
        if fingerprint != store.media["fingerprint"] and not allow_identity_change:
            raise MediaRelinkRequired(store.path, recorded, "fingerprint mismatch")
        store.relink_media(candidate, allow_identity_change=allow_identity_change)
        return store

    @classmethod
    def migrate_surgui_project(
        cls,
        project_path: Path,
        legacy_path: Path,
        media_path: Path,
        **media_info: Any
    ) -> "UnifiedVideoProjectStore":
        """Import the lossless temporal content of a SurgiAnnotate v1 project."""

        with Path(legacy_path).open("r", encoding="utf-8") as stream:
            legacy = json.load(stream)
        if not isinstance(legacy, dict) or not isinstance(legacy.get("media"), dict):
            raise ValueError("Legacy SurgiAnnotate project is invalid")
        store = cls.create(project_path, media_path, **media_info)
        store.project["temporal_templates"] = copy.deepcopy(
            legacy.get("templates", [])
        )
        migrated = []
        for raw in legacy.get("annotations", []):
            if not isinstance(raw, dict):
                continue
            annotation = TemporalAnnotation(
                id=str(raw.get("id") or make_id("time")),
                kind=str(raw.get("kind") or "point"),
                category=str(raw.get("category") or "event"),
                label=str(raw.get("label") or "Unnamed"),
                start_ms=int(raw.get("start_ms") or 0),
                end_ms=(
                    int(raw["end_ms"]) if raw.get("end_ms") is not None else None
                ),
                status=str(raw.get("status") or "draft"),
                note=str(raw.get("note") or ""),
                frame_index=(
                    int(raw["frame"]) if raw.get("frame") is not None else None
                ),
                score=(
                    int(raw["score"]) if raw.get("score") is not None else None
                ),
                asset_path=str(raw.get("asset_path") or ""),
                template_id=str(raw.get("template_id") or ""),
                created_at=str(raw.get("created_at") or utc_now()),
                updated_at=str(raw.get("updated_at") or utc_now()),
            )
            store._validate_temporal_annotation(
                annotation, int(store.media["duration_ms"])
            )
            migrated.append(asdict(annotation))
        store.project["temporal_annotations"] = migrated
        store.project["legacy_source"] = str(Path(legacy_path).resolve())
        # Keep source provenance intact instead of pretending that migration
        # created the historical records. The unified project has its own ID,
        # while the source identity and audit remain available for traceability.
        store.project["legacy_project"] = {
            "schema_version": legacy.get("schema_version"),
            "project_id": legacy.get("project_id"),
            "created_at": legacy.get("created_at"),
            "updated_at": legacy.get("updated_at"),
            "media": copy.deepcopy(legacy.get("media", {})),
            "audit": copy.deepcopy(legacy.get("audit", [])),
        }
        store._append_audit(
            "surgui_project_migrated",
            "Migrated {} video annotations from {}".format(
                len(migrated), Path(legacy_path).name
            ),
        )
        store.is_dirty = True
        return store

    @classmethod
    def load(cls, project_path: Path) -> "UnifiedVideoProjectStore":
        with project_write_lock(project_path):
            return cls._load_locked(project_path)

    @classmethod
    def _load_locked(cls, project_path: Path) -> "UnifiedVideoProjectStore":
        path = Path(project_path)
        try:
            store = cls._load_file(path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            backup = path.with_name("{}.previous{}".format(path.stem, path.suffix))
            if not backup.is_file():
                raise ValueError("Unable to load video project: {}".format(error)) from error
            try:
                store = cls._load_file(backup)
            except Exception as backup_error:
                raise ValueError(
                    "Video project and backup are invalid: {}; {}".format(
                        error, backup_error
                    )
                ) from error
            store.path = path
            store.recovered_from_backup = True
            store.recovery_source = backup
            store._append_audit("project_recovered", "Recovered from backup")
            store.is_dirty = True
        from IVAS.video.transactions import recover_label_transactions
        recover_label_transactions(store)
        return store

    @classmethod
    def _load_file(cls, path: Path) -> "UnifiedVideoProjectStore":
        with Path(path).open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("Project root must be an object")
        return cls(path, data)

    @property
    def media(self) -> Dict[str, Any]:
        return self.project["media"]

    @property
    def temporal_templates(self) -> List[TemporalTemplate]:
        return [
            TemporalTemplate.from_dict(item)
            for item in self.project["temporal_templates"]
        ]

    @property
    def temporal_annotations(self) -> List[TemporalAnnotation]:
        return [
            TemporalAnnotation.from_dict(item)
            for item in self.project["temporal_annotations"]
        ]

    @property
    def case_directory(self) -> Path:
        return self.path.parent

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @property
    def frames_directory(self) -> Path:
        return self.case_directory / "frames"

    @property
    def labels_directory(self) -> Path:
        return self.case_directory / "labels"

    @property
    def tracking_directory(self) -> Path:
        return self.case_directory / "tracking"

    def _normalise(self) -> None:
        version = int(self.project.get("schema_version", 0))
        if version == 2:
            self.project["schema_version"] = SCHEMA_VERSION
            self.project.setdefault("frame_map", {})
            self.project.setdefault("media_history", [])
        elif version != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported video project schema {}; expected {}".format(
                    version, SCHEMA_VERSION
                )
            )
        self.project.setdefault("app_version", APP_VERSION)
        self.project.setdefault("created_at", utc_now())
        self.project.setdefault("updated_at", self.project["created_at"])
        self.project.setdefault("project_id", make_id("case"))
        for key in (
            "temporal_annotations",
            "temporal_templates",
            "instances",
            "spatial_frames",
            "tracking_runs",
            "audit",
            "media_history",
        ):
            value = self.project.setdefault(key, [])
            if not isinstance(value, list):
                raise ValueError("Project field '{}' must be a list".format(key))
        frame_map = self.project.setdefault("frame_map", {})
        if not isinstance(frame_map, dict):
            raise ValueError("Project field 'frame_map' must be an object")

        media = self.project.get("media")
        if not isinstance(media, dict) or media.get("type") != "video":
            raise ValueError("Project media must describe a video")
        for key in ("duration_ms", "frame_count", "width", "height"):
            value = int(media.get(key, 0) or 0)
            if value < 0:
                raise ValueError("Media field '{}' cannot be negative".format(key))
            media[key] = value
        fps = float(media.get("fps", 0.0) or 0.0)
        if not math.isfinite(fps) or fps < 0:
            raise ValueError("Media fps must be a finite non-negative number")
        media["fps"] = round(fps, 6)
        for key in ("path", "file_name", "fingerprint"):
            if not isinstance(media.get(key), str) or not media[key]:
                raise ValueError("Media field '{}' is required".format(key))

        template_ids = set()
        normalised_templates = []
        for raw in self.project["temporal_templates"]:
            if not isinstance(raw, dict):
                raise ValueError("Temporal template must be an object")
            template = TemporalTemplate.from_dict(raw)
            template.id = str(template.id).strip()
            template.label = str(template.label).strip()
            template.kind = str(template.kind).strip().lower()
            template.category = str(template.category).strip()
            if not template.id or not template.label or not template.category:
                raise ValueError("Temporal template id, category and label are required")
            if template.kind not in VALID_TEMPORAL_KINDS:
                raise ValueError("Invalid temporal template kind: {}".format(template.kind))
            if template.id in template_ids:
                raise ValueError("Duplicate temporal template id: {}".format(template.id))
            if template.kind == "rating" and not template.score_options:
                template.score_options = [0, 1]
            template.score_options = sorted(set(int(item) for item in template.score_options))
            template_ids.add(template.id)
            normalised_templates.append(asdict(template))
        self.project["temporal_templates"] = normalised_templates

        annotation_ids = set()
        normalised_annotations = []
        duration_ms = media["duration_ms"]
        for raw in self.project["temporal_annotations"]:
            if not isinstance(raw, dict):
                raise ValueError("Video annotation must be an object")
            annotation = TemporalAnnotation.from_dict(raw)
            annotation.kind = str(annotation.kind).strip().lower()
            annotation.category = str(annotation.category).strip()
            annotation.label = str(annotation.label).strip()
            annotation.template_id = str(annotation.template_id or "").strip()
            if not annotation.template_id:
                matching = next(
                    (
                        item
                        for item in normalised_templates
                        if item["kind"] == annotation.kind
                        and item["category"].casefold() == annotation.category.casefold()
                        and item["label"].casefold() == annotation.label.casefold()
                    ),
                    None,
                )
                if matching is None:
                    template = TemporalTemplate(
                        id=make_id("tpl"),
                        label=annotation.label,
                        kind=annotation.kind,
                        category=annotation.category,
                        group="Migrated",
                        score_options=(
                            [annotation.score]
                            if annotation.kind == "rating" and annotation.score is not None
                            else []
                        ),
                        source="schema-v2-migration",
                    )
                    matching = asdict(template)
                    normalised_templates.append(matching)
                    template_ids.add(template.id)
                annotation.template_id = str(matching["id"])
            if annotation.template_id not in template_ids:
                raise ValueError(
                    "Video annotation references unknown template: {}".format(
                        annotation.template_id
                    )
                )
            self._validate_temporal_annotation(annotation, duration_ms)
            if annotation.id in annotation_ids:
                raise ValueError("Duplicate video annotation id: {}".format(annotation.id))
            annotation_ids.add(annotation.id)
            normalised_annotations.append(asdict(annotation))
        self.project["temporal_annotations"] = normalised_annotations
        self.project["temporal_templates"] = normalised_templates

        instance_ids = set()
        track_ids = set()
        normalised_instances = []
        for raw in self.project["instances"]:
            if not isinstance(raw, dict):
                raise ValueError("Tracked instance must be an object")
            item = dict(raw)
            object_id = str(item.get("object_id") or item.get("id") or make_id("obj"))
            track_id = str(item.get("track_id") or object_id)
            if object_id in instance_ids or track_id in track_ids:
                raise ValueError("Tracked instance identities must be unique")
            item.update(
                {
                    "object_id": object_id,
                    "track_id": track_id,
                    "category": str(item.get("category") or "unknown"),
                    "label": str(item.get("label") or item.get("category") or "unknown"),
                    "review_status": str(item.get("review_status") or "draft"),
                    "created_at": str(item.get("created_at") or utc_now()),
                    "updated_at": str(item.get("updated_at") or utc_now()),
                }
            )
            if item["review_status"] not in VALID_STATUSES:
                raise ValueError("Invalid instance review status")
            instance_ids.add(object_id)
            track_ids.add(track_id)
            normalised_instances.append(item)
        self.project["instances"] = normalised_instances

        spatial_indices = set()
        normalised_spatial = []
        for raw in self.project["spatial_frames"]:
            if not isinstance(raw, dict):
                raise ValueError("Spatial frame must be an object")
            item = dict(raw)
            frame_index = int(item.get("frame_index", -1))
            pts_ms = int(item.get("pts_ms", -1))
            if frame_index < 0 or pts_ms < 0:
                raise ValueError("Spatial frame index and timestamp cannot be negative")
            if frame_index in spatial_indices:
                raise ValueError("Duplicate spatial frame index: {}".format(frame_index))
            spatial_indices.add(frame_index)
            item["frame_index"] = frame_index
            item["pts_ms"] = pts_ms
            item.setdefault("image_path", "")
            item.setdefault("annotation_path", "")
            if "annotation_status" not in item:
                annotation_path = Path(str(item["annotation_path"]))
                if not annotation_path.is_absolute():
                    annotation_path = self.case_directory / annotation_path
                item["annotation_status"] = (
                    "saved" if annotation_path.is_file() else "pending"
                )
            if item["annotation_status"] not in {"pending", "saved"}:
                raise ValueError(
                    "Invalid spatial annotation status: {}".format(
                        item["annotation_status"]
                    )
                )
            item.setdefault("updated_at", utc_now())
            normalised_spatial.append(item)
        self.project["spatial_frames"] = normalised_spatial

        run_ids = set()
        normalised_runs = []
        for raw in self.project["tracking_runs"]:
            if not isinstance(raw, dict):
                raise ValueError("Tracking run must be an object")
            run = dict(raw)
            run_id = str(run.get("id") or make_id("trackrun"))
            status = str(run.get("status") or "queued")
            if run_id in run_ids:
                raise ValueError("Duplicate tracking run id: {}".format(run_id))
            if status not in VALID_TRACKING_STATUSES:
                raise ValueError("Invalid tracking run status: {}".format(status))
            direction = str(run.get("direction") or "forward")
            if direction not in {"forward", "backward", "bidirectional"}:
                raise ValueError("Invalid tracking direction: {}".format(direction))
            run.update(
                {
                    "id": run_id,
                    "status": status,
                    "direction": direction,
                    "start_frame": max(0, int(run.get("start_frame", 0))),
                    "end_frame": max(0, int(run.get("end_frame", 0))),
                    "instance_ids": [str(item) for item in run.get("instance_ids", [])],
                    "staged_frames": list(run.get("staged_frames", [])),
                    "created_at": str(run.get("created_at") or utc_now()),
                    "updated_at": str(run.get("updated_at") or utc_now()),
                }
            )
            run_ids.add(run_id)
            normalised_runs.append(run)
        self.project["tracking_runs"] = normalised_runs

    @staticmethod
    def _validate_temporal_annotation(
        annotation: TemporalAnnotation, duration_ms: int
    ) -> None:
        if not annotation.id or not annotation.label.strip() or not annotation.category.strip():
            raise ValueError("Video annotation id, category and label are required")
        if annotation.kind not in VALID_TEMPORAL_KINDS:
            raise ValueError("Invalid video annotation kind: {}".format(annotation.kind))
        if annotation.status not in VALID_STATUSES:
            raise ValueError("Invalid video annotation status: {}".format(annotation.status))
        annotation.start_ms = int(annotation.start_ms)
        if annotation.start_ms < 0:
            raise ValueError("Video annotation start cannot be negative")
        if annotation.end_ms is not None:
            annotation.end_ms = int(annotation.end_ms)
            if annotation.end_ms < annotation.start_ms:
                raise ValueError("Video annotation end cannot precede its start")
        if annotation.kind == "interval" and annotation.end_ms is None:
            raise ValueError("Interval annotation requires an end time")
        if annotation.kind != "interval" and annotation.end_ms is not None:
            raise ValueError("Point and rating annotations cannot have an end time")
        if annotation.kind == "rating":
            if annotation.score is None:
                raise ValueError("Rating annotation requires a score")
            annotation.score = int(annotation.score)
            annotation.start_ms = 0
            annotation.frame_index = None
        elif annotation.score is not None:
            raise ValueError("Only rating annotations can have a score")
        if annotation.frame_index is not None:
            annotation.frame_index = int(annotation.frame_index)
            if annotation.frame_index < 0:
                raise ValueError("Video annotation frame index cannot be negative")
        if duration_ms:
            if annotation.start_ms > duration_ms:
                raise ValueError("Video annotation starts after the video")
            if annotation.end_ms is not None and annotation.end_ms > duration_ms:
                raise ValueError("Video annotation ends after the video")

    def _snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.project)

    def _temporal_snapshot(self, project=None) -> Dict[str, Any]:
        source = self.project if project is None else project
        return {key: copy.deepcopy(source[key]) for key in
                ("temporal_annotations", "temporal_templates")}

    def _runtime_snapshot(self, deep_history: bool = True) -> Dict[str, Any]:
        """Capture all mutable in-memory state needed for an exact rollback."""

        return {
            "path": self.path,
            "project": self._snapshot(),
            "undo_stack": (
                copy.deepcopy(self._undo_stack)
                if deep_history
                else list(self._undo_stack)
            ),
            "redo_stack": (
                copy.deepcopy(self._redo_stack)
                if deep_history
                else list(self._redo_stack)
            ),
            "is_dirty": self.is_dirty,
            "recovered_from_backup": self.recovered_from_backup,
            "recovery_source": self.recovery_source,
            "persisted_project": copy.deepcopy(self._persisted_project),
        }

    def _restore_runtime_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        self.path = Path(snapshot["path"])
        self.project = copy.deepcopy(snapshot["project"])
        self._undo_stack = copy.deepcopy(snapshot["undo_stack"])
        self._redo_stack = copy.deepcopy(snapshot["redo_stack"])
        self.is_dirty = bool(snapshot["is_dirty"])
        self.recovered_from_backup = bool(snapshot["recovered_from_backup"])
        self.recovery_source = snapshot["recovery_source"]
        self._persisted_project = copy.deepcopy(snapshot["persisted_project"])

    @contextmanager
    def transaction(self) -> Iterator["UnifiedVideoProjectStore"]:
        """Rollback project and history state if a grouped change fails.

        Callers can wrap a mutation followed immediately by :meth:`save` in
        this context.  If persistence raises, the exception is re-raised after
        the complete in-memory state has been restored.
        """

        snapshot = self._runtime_snapshot()
        try:
            yield self
        except BaseException:
            self._restore_runtime_snapshot(snapshot)
            raise

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        """Make one public mutator atomic, including its history bookkeeping."""

        # A normal mutator only appends/clears history lists; it never edits
        # snapshots already in them.  A shallow list copy therefore preserves
        # exact rollback without repeatedly cloning the full project history.
        snapshot = self._runtime_snapshot(deep_history=False)
        try:
            yield
            before = self._temporal_snapshot(snapshot["project"])
            if before != self._temporal_snapshot():
                self._undo_stack.append(before)
                self._redo_stack.clear()
        except BaseException:
            self._restore_runtime_snapshot(snapshot)
            raise

    def _append_audit(self, action: str, detail: str, target_id: str = "") -> None:
        self.project["audit"].append(
            {
                "id": make_id("audit"),
                "timestamp": utc_now(),
                "action": action,
                "detail": detail,
                "target_id": target_id,
            }
        )
        self.project["updated_at"] = utc_now()

    def _changed(self, action: str, detail: str, target_id: str = "") -> None:
        self._append_audit(action, detail, target_id)
        self.is_dirty = True

    def _update_media_path(self, media_path: Path) -> None:
        resolved = Path(media_path).resolve()
        updates = {
            "path": str(resolved),
            "file_name": resolved.name,
        }
        if all(self.media.get(key) == value for key, value in updates.items()):
            return
        with self._mutation():
            self.media.update(updates)
            self._normalise()
            self._changed("media_path_updated", "Updated video media path")

    def relink_media(
        self, media_path: Path, allow_identity_change: bool = False
    ) -> None:
        """Explicitly relink moved media, preserving the prior binding in history."""

        replacement_path = Path(media_path).resolve()
        if not replacement_path.is_file():
            raise ValueError("Video file does not exist: {}".format(replacement_path))
        fingerprint = media_fingerprint(replacement_path)
        identity_changed = fingerprint != self.media["fingerprint"]
        if identity_changed and not allow_identity_change:
            raise ValueError(
                "Replacement media fingerprint differs; explicit identity-change consent is required"
            )
        replacement = {
            "path": str(replacement_path),
            "file_name": replacement_path.name,
            "fingerprint": fingerprint,
        }
        if all(self.media.get(key) == value for key, value in replacement.items()):
            return
        with self._mutation():
            self.project["media_history"].append(
                {
                    "path": self.media.get("path", ""),
                    "file_name": self.media.get("file_name", ""),
                    "fingerprint": self.media.get("fingerprint", ""),
                    "replaced_at": utc_now(),
                    "identity_changed": identity_changed,
                }
            )
            self.media.update(replacement)
            self._normalise()
            self._changed(
                "media_identity_changed" if identity_changed else "media_relinked",
                "Explicitly relinked video media",
            )

    def update_media_info(self, **changes: Any) -> None:
        accepted = {"duration_ms", "fps", "frame_count", "width", "height"}
        updates: Dict[str, Any] = {}
        for key, value in changes.items():
            if key not in accepted or value is None:
                continue
            if key == "fps":
                normalised = float(value)
                if not math.isfinite(normalised):
                    raise ValueError("Media fps must be a finite non-negative number")
                normalised = round(max(0.0, normalised), 6)
            else:
                normalised = max(0, int(value))
            if key == "duration_ms" and self.project["temporal_annotations"]:
                last_end = max(
                    int(item.get("end_ms") or item.get("start_ms") or 0)
                    for item in self.project["temporal_annotations"]
                )
                normalised = max(normalised, last_end)
            if self.media.get(key) != normalised:
                updates[key] = normalised
        if updates:
            with self._mutation():
                self.media.update(updates)
                self._normalise()
                self._changed("media_info_updated", "Updated decoded video metadata")

    def set_frame_map(
        self,
        asset_path: Path,
        frame_count: int,
        duration_ms: int,
        method: str,
        fingerprint: str = "",
    ) -> Dict[str, Any]:
        path = Path(asset_path).resolve()
        try:
            stored_path = path.relative_to(self.case_directory.resolve()).as_posix()
        except ValueError:
            stored_path = str(path)
        frame_map = {
            "asset_path": stored_path,
            "frame_count": max(0, int(frame_count)),
            "duration_ms": max(0, int(duration_ms)),
            "method": str(method),
            "media_fingerprint": str(fingerprint or self.media["fingerprint"]),
            "updated_at": utc_now(),
        }
        if frame_map == self.project.get("frame_map"):
            return dict(frame_map)
        with self._mutation():
            self.project["frame_map"] = frame_map
            self._changed("frame_map_updated", "Updated decoded presentation timestamp map")
        return dict(frame_map)

    def _template_map(self) -> Dict[str, TemporalTemplate]:
        return {template.id: template for template in self.temporal_templates}

    def _annotation_index(self, annotation_id: str) -> int:
        for index, raw in enumerate(self.project["temporal_annotations"]):
            if raw.get("id") == annotation_id:
                return index
        raise KeyError("Unknown video annotation: {}".format(annotation_id))

    def add_template(self, template: TemporalTemplate) -> TemporalTemplate:
        item = TemporalTemplate.from_dict(asdict(template))
        item.id = str(item.id or make_id("tpl")).strip()
        item.label = str(item.label).strip()
        item.kind = str(item.kind).strip().lower()
        item.category = str(item.category).strip()
        if not item.id or not item.label or not item.category:
            raise ValueError("Template id, category and label are required")
        if item.kind not in VALID_TEMPORAL_KINDS:
            raise ValueError("Invalid temporal template kind: {}".format(item.kind))
        if item.id in self._template_map():
            raise ValueError("Duplicate temporal template id: {}".format(item.id))
        if item.kind == "rating" and not item.score_options:
            item.score_options = [0, 1]
        with self._mutation():
            self.project["temporal_templates"].append(asdict(item))
            self._normalise()
            self._changed("temporal_template_added", "Added template '{}'".format(item.label), item.id)
        return item

    def append_templates(
        self, templates: Sequence[TemporalTemplate]
    ) -> List[TemporalTemplate]:
        if not templates:
            return []
        candidates = [TemporalTemplate.from_dict(asdict(item)) for item in templates]
        existing = set(self._template_map())
        identifiers = [item.id for item in candidates]
        if len(set(identifiers)) != len(identifiers) or any(
            identifier in existing for identifier in identifiers
        ):
            raise ValueError("Imported templates contain duplicate IDs")
        with self._mutation():
            self.project["temporal_templates"].extend(asdict(item) for item in candidates)
            self._normalise()
            self._changed(
                "temporal_templates_added",
                "Added {} temporal templates".format(len(candidates)),
            )
        return candidates

    def update_template(self, template_id: str, **changes: Any) -> TemporalTemplate:
        permitted = {
            "label",
            "category",
            "group",
            "colour",
            "start_hint",
            "end_hint",
            "score_options",
            "required",
        }
        unexpected = set(changes) - permitted
        if unexpected:
            raise ValueError("Unsupported template fields: {}".format(", ".join(sorted(unexpected))))
        index = next(
            (
                index
                for index, raw in enumerate(self.project["temporal_templates"])
                if raw.get("id") == template_id
            ),
            None,
        )
        if index is None:
            raise KeyError("Unknown temporal template: {}".format(template_id))
        current = TemporalTemplate.from_dict(self.project["temporal_templates"][index])
        data = asdict(current)
        data.update(changes)
        updated = TemporalTemplate.from_dict(data)
        updated.label = updated.label.strip()
        updated.category = updated.category.strip()
        if not updated.label or not updated.category:
            raise ValueError("Template category and label are required")
        if updated == current:
            return current
        with self._mutation():
            self.project["temporal_templates"][index] = asdict(updated)
            timestamp = utc_now()
            for raw in self.project["temporal_annotations"]:
                if raw.get("template_id") == template_id:
                    raw["label"] = updated.label
                    raw["category"] = updated.category
                    raw["updated_at"] = timestamp
            self._normalise()
            self._changed("temporal_template_updated", "Updated template '{}'".format(updated.label), template_id)
        return updated

    def delete_template(self, template_id: str) -> TemporalTemplate:
        index = next(
            (
                index
                for index, raw in enumerate(self.project["temporal_templates"])
                if raw.get("id") == template_id
            ),
            None,
        )
        if index is None:
            raise KeyError("Unknown temporal template: {}".format(template_id))
        references = sum(
            raw.get("template_id") == template_id
            for raw in self.project["temporal_annotations"]
        )
        if references:
            raise ValueError(
                "Template is referenced by {} annotations; reassign or delete them first".format(
                    references
                )
            )
        with self._mutation():
            deleted = TemporalTemplate.from_dict(
                self.project["temporal_templates"].pop(index)
            )
            self._changed("temporal_template_deleted", "Deleted template '{}'".format(deleted.label), template_id)
        return deleted

    def _matching_template(
        self, kind: str, category: str, label: str
    ) -> Optional[TemporalTemplate]:
        return next(
            (
                template
                for template in self.temporal_templates
                if template.kind == kind
                and template.category.casefold() == str(category).casefold()
                and template.label.casefold() == str(label).casefold()
            ),
            None,
        )

    def ensure_template(
        self,
        kind: str,
        category: str,
        label: str,
        score_options: Optional[Sequence[int]] = None,
    ) -> TemporalTemplate:
        existing = self._matching_template(kind, category, label)
        if existing is not None:
            return existing
        return self.add_template(
            TemporalTemplate(
                id=make_id("tpl"),
                label=str(label).strip(),
                kind=str(kind).strip().lower(),
                category=str(category).strip(),
                score_options=[int(value) for value in (score_options or [])],
            )
        )

    def add_temporal_annotation(
        self,
        template_id: str,
        start_ms: int = 0,
        end_ms: Optional[int] = None,
        note: str = "",
        status: str = "draft",
        frame_index: Optional[int] = None,
        score: Optional[int] = None,
        asset_path: str = "",
    ) -> TemporalAnnotation:
        template = self._template_map().get(str(template_id))
        if template is None:
            raise ValueError("Template does not belong to this project")
        if template.kind == "rating":
            if score is None or int(score) not in template.score_options:
                raise ValueError("Rating score must be one of {}".format(template.score_options))
            start_ms = 0
            end_ms = None
            frame_index = None
        elif template.kind == "point":
            end_ms = None
        candidate = TemporalAnnotation(
            id=make_id("time"),
            kind=template.kind,
            category=template.category,
            label=template.label,
            start_ms=int(start_ms),
            end_ms=int(end_ms) if end_ms is not None else None,
            status=str(status),
            note=str(note or ""),
            frame_index=int(frame_index) if frame_index is not None else None,
            score=int(score) if score is not None else None,
            asset_path=str(asset_path or ""),
            template_id=template.id,
        )
        self._validate_temporal_annotation(candidate, self.media["duration_ms"])
        with self._mutation():
            if candidate.kind == "rating":
                self.project["temporal_annotations"] = [
                    raw
                    for raw in self.project["temporal_annotations"]
                    if not (
                        raw.get("kind") == "rating"
                        and raw.get("template_id") == candidate.template_id
                    )
                ]
            self.project["temporal_annotations"].append(asdict(candidate))
            self._changed(
                "temporal_annotation_added",
                "Added {} annotation '{}'".format(candidate.kind, candidate.label),
                candidate.id,
            )
        return candidate

    def update_temporal_annotation(
        self, annotation_id: str, **changes: Any
    ) -> TemporalAnnotation:
        permitted = {
            "template_id",
            "start_ms",
            "end_ms",
            "note",
            "status",
            "frame_index",
            "score",
            "asset_path",
        }
        unexpected = set(changes) - permitted
        if unexpected:
            raise ValueError("Unsupported annotation fields: {}".format(", ".join(sorted(unexpected))))
        index = self._annotation_index(annotation_id)
        current = TemporalAnnotation.from_dict(
            self.project["temporal_annotations"][index]
        )
        template_id = str(changes.get("template_id", current.template_id))
        template = self._template_map().get(template_id)
        if template is None:
            raise ValueError("Target template does not belong to this project")
        if template.kind != current.kind:
            raise ValueError("Annotation kind cannot be changed during editing")
        data = asdict(current)
        data.update(changes)
        data.update(
            {
                "template_id": template.id,
                "label": template.label,
                "category": template.category,
                "updated_at": utc_now(),
            }
        )
        updated = TemporalAnnotation.from_dict(data)
        if template.kind == "rating":
            if updated.score not in template.score_options:
                raise ValueError("Rating score must be one of {}".format(template.score_options))
            updated.start_ms = 0
            updated.end_ms = None
            updated.frame_index = None
        elif template.kind == "point":
            updated.end_ms = None
            updated.score = None
        else:
            updated.score = None
        self._validate_temporal_annotation(updated, self.media["duration_ms"])
        if updated == current:
            return current
        with self._mutation():
            self.project["temporal_annotations"][index] = asdict(updated)
            self._changed("temporal_annotation_updated", "Updated '{}'".format(updated.label), updated.id)
        return updated

    def add_interval(
        self,
        category: str,
        label: str,
        start_ms: int,
        end_ms: int,
        note: str = "",
        status: str = "draft",
    ) -> TemporalAnnotation:
        template = self.ensure_template("interval", category, label)
        return self.add_temporal_annotation(
            template.id,
            start_ms=start_ms,
            end_ms=end_ms,
            note=note,
            status=status,
        )

    def add_point(
        self,
        template_id: str,
        start_ms: int,
        frame_index: Optional[int] = None,
        note: str = "",
        status: str = "draft",
        asset_path: str = "",
    ) -> TemporalAnnotation:
        return self.add_temporal_annotation(
            template_id,
            start_ms=start_ms,
            frame_index=frame_index,
            note=note,
            status=status,
            asset_path=asset_path,
        )

    def set_rating(
        self,
        template_id: str,
        score: int,
        note: str = "",
        status: str = "draft",
    ) -> TemporalAnnotation:
        return self.add_temporal_annotation(
            template_id,
            score=score,
            note=note,
            status=status,
        )

    def delete_temporal_annotation(self, annotation_id: str) -> TemporalAnnotation:
        for index, raw in enumerate(self.project["temporal_annotations"]):
            if raw.get("id") == annotation_id:
                annotation = TemporalAnnotation.from_dict(raw)
                with self._mutation():
                    del self.project["temporal_annotations"][index]
                    self._changed(
                        "temporal_annotation_deleted",
                        "Deleted interval '{}'".format(annotation.label),
                        annotation.id,
                    )
                return annotation
        raise KeyError("Unknown video annotation: {}".format(annotation_id))

    def add_spatial_frame(
        self,
        frame_index: int,
        pts_ms: int,
        image_path: Path,
        annotation_path: Path,
        annotation_status: str = "pending",
    ) -> Dict[str, Any]:
        frame_index = int(frame_index)
        pts_ms = int(pts_ms)
        if frame_index < 0 or pts_ms < 0:
            raise ValueError("Spatial frame index and timestamp cannot be negative")
        annotation_status = str(annotation_status)
        if annotation_status not in {"pending", "saved"}:
            raise ValueError(
                "Invalid spatial annotation status: {}".format(annotation_status)
            )

        def stored_path(value: Path) -> str:
            path = Path(value).resolve()
            try:
                return path.relative_to(self.case_directory.resolve()).as_posix()
            except ValueError:
                return str(path)

        item = {
            "frame_index": frame_index,
            "pts_ms": pts_ms,
            "image_path": stored_path(Path(image_path)),
            "annotation_path": stored_path(Path(annotation_path)),
            "annotation_status": annotation_status,
            "updated_at": utc_now(),
        }
        with self._mutation():
            replaced = False
            for index, existing in enumerate(self.project["spatial_frames"]):
                if int(existing["frame_index"]) == frame_index:
                    self.project["spatial_frames"][index] = item
                    replaced = True
                    break
            if not replaced:
                self.project["spatial_frames"].append(item)
                self.project["spatial_frames"].sort(key=lambda value: value["frame_index"])
            self._changed(
                "spatial_frame_linked",
                "Linked frame {} to an IVAS annotation".format(frame_index),
                "frame_{}".format(frame_index),
            )
        return dict(item)

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        annotations = self.temporal_annotations
        templates = self._template_map()
        for annotation in annotations:
            template = templates.get(annotation.template_id)
            if template is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "Annotation references a missing template",
                        annotation.id,
                    )
                )
                continue
            if annotation.status == "draft":
                issues.append(
                    ValidationIssue(
                        "info",
                        "'{}' is still draft".format(annotation.label),
                        annotation.id,
                    )
                )
            if template.required and annotation.kind == "rating" and annotation.score is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "Required rating '{}' has no score".format(annotation.label),
                        annotation.id,
                    )
                )

        present_templates = {item.template_id for item in annotations}
        for template in templates.values():
            if template.required and template.id not in present_templates:
                issues.append(
                    ValidationIssue(
                        "error", "Required template '{}' has no annotation".format(template.label)
                    )
                )

        interval_groups: Dict[Tuple[str, str], List[TemporalAnnotation]] = {}
        for annotation in annotations:
            if annotation.kind == "interval":
                interval_groups.setdefault(
                    (annotation.category.casefold(), annotation.template_id), []
                ).append(annotation)
        for repeated in interval_groups.values():
            repeated.sort(key=lambda item: (item.start_ms, item.end_ms or item.start_ms))
            previous: Optional[TemporalAnnotation] = None
            for annotation in repeated:
                if previous and annotation.start_ms < int(previous.end_ms or previous.start_ms):
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "Overlapping '{}' intervals".format(annotation.label),
                            annotation.id,
                        )
                    )
                if previous is None or int(annotation.end_ms or 0) > int(previous.end_ms or 0):
                    previous = annotation
        return issues

    def phase_coverage_ratio(self) -> float:
        duration = int(self.media.get("duration_ms", 0))
        if duration <= 0:
            return 0.0
        phases = [
            annotation
            for annotation in self.temporal_annotations
            if annotation.kind == "interval"
            and annotation.category.casefold() == "phase"
        ]
        return min(1.0, interval_coverage_ms(phases, duration) / float(duration))

    def export_csv(self, output_path: Path) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "project_id",
            "video_file",
            "annotation_id",
            "kind",
            "category",
            "template_id",
            "label",
            "start_ms",
            "end_ms",
            "start_timecode",
            "end_timecode",
            "frame_index",
            "score",
            "status",
            "note",
            "asset_path",
            "created_at",
            "updated_at",
        )
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for annotation in sorted(
                self.temporal_annotations,
                key=lambda item: (item.start_ms, item.created_at, item.id),
            ):
                row = asdict(annotation)
                row.update(
                    {
                        "project_id": self.project["project_id"],
                        "video_file": self.media.get("file_name", ""),
                        "annotation_id": annotation.id,
                        "start_timecode": format_timecode(annotation.start_ms),
                        "end_timecode": format_timecode(annotation.end_ms),
                    }
                )
                row.pop("id", None)
                writer.writerow(row)
        return destination

    def import_csv(self, input_path: Path) -> List[ValidationIssue]:
        source = Path(input_path)
        if not source.is_file():
            raise ValueError("CSV file does not exist: {}".format(source))
        templates = self._template_map()
        staged = copy.deepcopy(self.project["temporal_annotations"])
        identities = {
            (
                raw.get("kind"),
                raw.get("template_id"),
                int(raw.get("start_ms") or 0),
                raw.get("end_ms"),
                raw.get("score"),
            )
            for raw in staged
        }
        reports: List[ValidationIssue] = []
        imported = 0
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            for line_number, row in enumerate(reader, start=2):
                try:
                    kind = str(row.get("kind") or "interval").strip().lower()
                    category = str(row.get("category") or "phase").strip()
                    label = str(row.get("label") or "").strip()
                    template_id = str(row.get("template_id") or "").strip()
                    template = templates.get(template_id)
                    if template is None:
                        template = next(
                            (
                                item
                                for item in templates.values()
                                if item.kind == kind
                                and item.category.casefold() == category.casefold()
                                and item.label.casefold() == label.casefold()
                            ),
                            None,
                        )
                    if template is None:
                        raise ValueError("No matching project template")
                    start_text = row.get("start_ms") or row.get("start_timecode") or "0"
                    start_ms = parse_timecode(start_text)
                    raw_end = row.get("end_ms") or row.get("end_timecode") or ""
                    end_ms = parse_timecode(raw_end) if raw_end and kind == "interval" else None
                    raw_score = str(row.get("score") or "").strip()
                    score = int(raw_score) if raw_score else None
                    raw_frame = str(row.get("frame_index") or "").strip()
                    frame_index = int(raw_frame) if raw_frame else None
                    candidate = TemporalAnnotation(
                        id=str(row.get("annotation_id") or make_id("time")),
                        kind=template.kind,
                        category=template.category,
                        label=template.label,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        status=str(row.get("status") or "draft"),
                        note=str(row.get("note") or ""),
                        frame_index=frame_index,
                        score=score,
                        asset_path=str(row.get("asset_path") or ""),
                        template_id=template.id,
                        created_at=str(row.get("created_at") or utc_now()),
                        updated_at=str(row.get("updated_at") or utc_now()),
                    )
                    self._validate_temporal_annotation(candidate, self.media["duration_ms"])
                except (TypeError, ValueError) as error:
                    reports.append(
                        ValidationIssue(
                            "warning", "{}:{} {}".format(source.name, line_number, error)
                        )
                    )
                    continue
                identity = (
                    candidate.kind,
                    candidate.template_id,
                    candidate.start_ms,
                    candidate.end_ms,
                    candidate.score,
                )
                if identity in identities:
                    reports.append(
                        ValidationIssue(
                            "warning",
                            "{}:{} duplicates an existing annotation".format(
                                source.name, line_number
                            ),
                        )
                    )
                    continue
                if any(raw.get("id") == candidate.id for raw in staged):
                    candidate.id = make_id("time")
                staged.append(asdict(candidate))
                identities.add(identity)
                imported += 1
        if imported:
            with self._mutation():
                self.project["temporal_annotations"] = staged
                self._normalise()
                self._changed("temporal_csv_imported", "Imported {} CSV annotations".format(imported))
        return reports

    def export_legacy(self, output_directory: Path) -> List[Path]:
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        used_names: Dict[str, str] = {}
        manifest_entries: List[Dict[str, str]] = []

        def target(category: str, suffix: str = "") -> Path:
            base = safe_filename(category)
            filename = "{}{}.txt".format(base, suffix)
            owner = used_names.get(filename.casefold())
            if owner and owner != category:
                filename = "{}_{}{}.txt".format(
                    base,
                    hashlib.sha256(category.encode("utf-8")).hexdigest()[:8],
                    suffix,
                )
            used_names[filename.casefold()] = category
            return destination / filename

        annotations = self.temporal_annotations
        categories = sorted(
            {item.category for item in annotations if item.kind in {"interval", "point"}}
        )
        for category in categories:
            path = target(category)
            rows = [
                item
                for item in annotations
                if item.category == category and item.kind in {"interval", "point"}
            ]
            with path.open("w", encoding="utf-8") as stream:
                for item in sorted(rows, key=lambda value: (value.start_ms, value.label, value.id)):
                    end_ms = item.end_ms if item.end_ms is not None else 0
                    stream.write(
                        "{} : ({},{})\n".format(
                            item.label,
                            format_legacy_timecode(item.start_ms),
                            format_legacy_timecode(end_ms),
                        )
                    )
            written.append(path)
            manifest_entries.append(
                {"file": path.name, "category": category, "kind": "interval"}
            )

        rating_categories: Dict[str, List[TemporalTemplate]] = {}
        for template in self.temporal_templates:
            if template.kind == "rating":
                rating_categories.setdefault(template.category, []).append(template)
        for category, templates in sorted(rating_categories.items()):
            path = target(category, "_scores")
            with path.open("w", encoding="utf-8") as stream:
                for template in templates:
                    rating = next(
                        (
                            item
                            for item in annotations
                            if item.kind == "rating" and item.template_id == template.id
                        ),
                        None,
                    )
                    if rating is None:
                        stream.write("{}\n".format(template.label))
                    else:
                        stream.write("{} : {}\n".format(template.label, rating.score))
            written.append(path)
            manifest_entries.append(
                {"file": path.name, "category": category, "kind": "rating"}
            )
        manifest_path = destination / "surgui_legacy_manifest.json"
        manifest_path.write_text(
            json.dumps({"schema_version": 1, "files": manifest_entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return written

    def sync_default_exports(self) -> Tuple[Path, List[Path]]:
        """Refresh the case-local CSV and legacy views after a project save.

        ``project.ivas.json`` remains the lossless source of truth.  These
        generated files are deliberately kept in a separate ``exports``
        directory so downstream review/training tools always have an up-to-
        date interchange view without requiring a final manual export step.
        """

        destination = self.case_directory / "exports"
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / "surgui_legacy_manifest.json"

        # Remove only filenames recorded by our previous manifest.  This
        # clears obsolete category exports while preserving unrelated files a
        # user may have placed beside the generated views.
        if manifest.is_file():
            try:
                previous = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                previous = {}
            if not isinstance(previous, dict):
                previous = {}
            recorded_files = previous.get("files", [])
            if not isinstance(recorded_files, list):
                recorded_files = []
            for entry in recorded_files:
                filename = str(entry.get("file") or "") if isinstance(entry, dict) else ""
                if not filename or Path(filename).name != filename:
                    continue
                candidate = destination / filename
                if candidate.is_file():
                    try:
                        candidate.unlink()
                    except OSError:
                        pass

        csv_path = self.export_csv(destination / "annotations.csv")
        legacy_paths = self.export_legacy(destination)
        return csv_path, legacy_paths

    def import_legacy(self, input_directory: Path) -> List[ValidationIssue]:
        directory = Path(input_directory)
        if not directory.is_dir():
            raise ValueError("Legacy annotation directory does not exist: {}".format(directory))
        reports: List[ValidationIssue] = []
        staged = copy.deepcopy(self.project["temporal_annotations"])
        templates = self.temporal_templates
        interval_pattern = re.compile(r"^\s*(.*?)\s*:\s*\(([^,]+),([^)]*)\)\s*$")
        imported = 0
        for path in sorted(directory.glob("*.txt"), key=lambda item: item.name.casefold()):
            category = path.stem[:-7] if path.stem.endswith("_scores") else path.stem
            with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
                for line_number, raw in enumerate(stream, start=1):
                    text = raw.strip()
                    if not text:
                        continue
                    if path.stem.endswith("_scores"):
                        if " : " not in text:
                            continue
                        label, raw_score = text.rsplit(" : ", 1)
                        template = next(
                            (
                                item
                                for item in templates
                                if item.kind == "rating"
                                and item.category.casefold() == category.casefold()
                                and item.label.casefold() == label.strip().casefold()
                            ),
                            None,
                        )
                        if template is None:
                            reports.append(ValidationIssue("warning", "{}:{} no matching rating template".format(path.name, line_number)))
                            continue
                        try:
                            score = int(raw_score)
                            if score not in template.score_options:
                                raise ValueError("score is outside template options")
                        except ValueError as error:
                            reports.append(ValidationIssue("warning", "{}:{} {}".format(path.name, line_number, error)))
                            continue
                        staged = [
                            item
                            for item in staged
                            if not (
                                item.get("kind") == "rating"
                                and item.get("template_id") == template.id
                            )
                        ]
                        staged.append(
                            asdict(
                                TemporalAnnotation(
                                    id=make_id("time"),
                                    kind="rating",
                                    category=template.category,
                                    label=template.label,
                                    start_ms=0,
                                    score=score,
                                    template_id=template.id,
                                )
                            )
                        )
                        imported += 1
                        continue
                    match = interval_pattern.match(text)
                    if match is None:
                        reports.append(ValidationIssue("warning", "{}:{} unrecognised line".format(path.name, line_number)))
                        continue
                    label, raw_start, raw_end = match.groups()
                    candidates = [
                        item
                        for item in templates
                        if item.kind in {"interval", "point"}
                        and item.category.casefold() == category.casefold()
                        and item.label.casefold() == label.strip().casefold()
                    ]
                    if len(candidates) != 1:
                        reports.append(ValidationIssue("warning", "{}:{} template match is missing or ambiguous".format(path.name, line_number)))
                        continue
                    template = candidates[0]
                    try:
                        annotation = TemporalAnnotation(
                            id=make_id("time"),
                            kind=template.kind,
                            category=template.category,
                            label=template.label,
                            start_ms=parse_timecode(raw_start),
                            end_ms=(parse_timecode(raw_end) if template.kind == "interval" else None),
                            template_id=template.id,
                        )
                        self._validate_temporal_annotation(annotation, self.media["duration_ms"])
                    except ValueError as error:
                        reports.append(ValidationIssue("warning", "{}:{} {}".format(path.name, line_number, error)))
                        continue
                    staged.append(asdict(annotation))
                    imported += 1
        if imported:
            with self._mutation():
                self.project["temporal_annotations"] = staged
                self._normalise()
                self._changed("temporal_legacy_imported", "Imported {} legacy annotations".format(imported))
        return reports

    def ensure_instance(
        self,
        category: str,
        label: str = "",
        object_id: str = "",
        track_id: str = "",
        review_status: str = "draft",
    ) -> Dict[str, Any]:
        candidate_object_id = str(object_id or make_id("obj"))
        candidate_track_id = str(track_id or candidate_object_id)
        for raw in self.project["instances"]:
            if raw.get("object_id") == candidate_object_id or raw.get("track_id") == candidate_track_id:
                if review_status == "reviewed" and raw.get("review_status") != "reviewed":
                    raw["review_status"] = "reviewed"
                    raw["updated_at"] = utc_now()
                    self._changed("tracking_instance_reviewed", "Reviewed tracked instance", raw["object_id"])
                return dict(raw)
        item = {
            "object_id": candidate_object_id,
            "track_id": candidate_track_id,
            "category": str(category or "unknown"),
            "label": str(label or category or "unknown"),
            "review_status": str(review_status),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        with self._mutation():
            self.project["instances"].append(item)
            self._normalise()
            self._changed("tracking_instance_added", "Added stable tracked instance", candidate_object_id)
        return dict(item)

    def create_tracking_run(
        self,
        direction: str,
        start_frame: int,
        end_frame: int,
        instance_ids: Sequence[str],
        engine: str,
        replace_manual: bool = False,
    ) -> Dict[str, Any]:
        direction = str(direction)
        if direction not in {"forward", "backward", "bidirectional"}:
            raise ValueError("Invalid tracking direction: {}".format(direction))
        start_frame = max(0, int(start_frame))
        end_frame = max(0, int(end_frame))
        if end_frame < start_frame:
            raise ValueError("Tracking end frame cannot precede its start")
        known = {item["object_id"] for item in self.project["instances"]}
        requested = [str(item) for item in instance_ids]
        if not requested or any(item not in known for item in requested):
            raise ValueError("Tracking run references unknown or empty instances")
        run = {
            "id": make_id("trackrun"),
            "status": "queued",
            "direction": direction,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "instance_ids": requested,
            "engine": str(engine),
            "replace_manual": bool(replace_manual),
            "staged_frames": [],
            "error": "",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        with self._mutation():
            self.project["tracking_runs"].append(run)
            self._changed("tracking_run_created", "Created staged tracking run", run["id"])
        return copy.deepcopy(run)

    def update_tracking_run(self, run_id: str, **changes: Any) -> Dict[str, Any]:
        permitted = {"status", "staged_frames", "error", "metrics", "committed_at", "rolled_back_at"}
        unexpected = set(changes) - permitted
        if unexpected:
            raise ValueError("Unsupported tracking run fields: {}".format(", ".join(sorted(unexpected))))
        index = next(
            (
                index
                for index, raw in enumerate(self.project["tracking_runs"])
                if raw.get("id") == run_id
            ),
            None,
        )
        if index is None:
            raise KeyError("Unknown tracking run: {}".format(run_id))
        updated = copy.deepcopy(self.project["tracking_runs"][index])
        updated.update(copy.deepcopy(changes))
        updated["updated_at"] = utc_now()
        if updated.get("status") not in VALID_TRACKING_STATUSES:
            raise ValueError("Invalid tracking run status")
        with self._mutation():
            self.project["tracking_runs"][index] = updated
            self._normalise()
            self._changed("tracking_run_updated", "Tracking run is {}".format(updated["status"]), run_id)
        return copy.deepcopy(updated)

    def undo(self) -> bool:
        """Undo video annotations/templates; retain media, frames and tracking."""
        if not self._undo_stack:
            return False
        snapshot = self._runtime_snapshot()
        try:
            self._redo_stack.append(self._temporal_snapshot())
            self.project.update(self._undo_stack.pop())
            self._append_audit("undo", "Undid the previous video annotation change")
            self.is_dirty = True
        except BaseException:
            self._restore_runtime_snapshot(snapshot)
            raise
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        snapshot = self._runtime_snapshot()
        try:
            self._undo_stack.append(self._temporal_snapshot())
            self.project.update(self._redo_stack.pop())
            self._append_audit("redo", "Redid the previous video annotation change")
            self.is_dirty = True
        except BaseException:
            self._restore_runtime_snapshot(snapshot)
            raise
        return True

    def save(self) -> None:
        with project_write_lock(self.path):
            self._save_locked()

    def refresh(self) -> None:
        """Accept disk updates without discarding this editor's temporal history."""
        with project_write_lock(self.path):
            remote = self._load_file(self.path)._persisted_project
            merged = merge_projects(self._persisted_project, self.project, remote)
            self.project = merged
            self._normalise()
            self._persisted_project = copy.deepcopy(remote)
            self.is_dirty = self.project != remote

    def _save_locked(self) -> None:
        if not self.is_dirty and self.path.is_file():
            return
        original = self.project
        self.project = copy.deepcopy(original)
        try:
            self._normalise()
            if self.path.is_file():
                try:
                    remote = self._load_file(self.path)._persisted_project
                except (OSError, ValueError, KeyError, TypeError):
                    if not self.recovered_from_backup:
                        raise
                    remote = self._persisted_project
                if self._persisted_project is None:
                    raise ProjectConflictError("A project already exists at {}".format(self.path))
                self.project = merge_projects(self._persisted_project, self.project, remote)
            elif self._persisted_project is not None and self._persisted_project.get("revision"):
                raise ProjectConflictError("The project was removed by another writer")
            self._normalise()
            self.project["revision"] = int(self.project.get("revision", 0)) + 1
            self.frames_directory.mkdir(parents=True, exist_ok=True)
            self.labels_directory.mkdir(parents=True, exist_ok=True)
            self.tracking_directory.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.project, ensure_ascii=False, indent=2).encode("utf-8")
            backup = self.path.with_name("{}.previous{}".format(self.path.stem, self.path.suffix))
            preserve_backup = bool(self.recovered_from_backup and self.recovery_source
                                   and self.recovery_source.resolve() == backup.resolve())
            if self.path.exists() and not preserve_backup:
                atomic_write_bytes(backup, self.path.read_bytes())
            atomic_write_bytes(self.path, payload)
        except BaseException:
            self.project = original
            raise
        self.is_dirty = False
        self.recovered_from_backup = False
        self.recovery_source = None
        self._persisted_project = copy.deepcopy(self.project)
