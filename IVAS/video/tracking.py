"""Staged, cancellable and transactional video object tracking services."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from IVAS.annotation import Annotation, Object
from IVAS.video.media import FrameTimestampMap, OperationCancelled, atomic_write_png
from IVAS.video.project import UnifiedVideoProjectStore, utc_now


@dataclass
class TrackingSeed:
    frame_index: int
    object_id: str
    mask_path: Path


@dataclass
class TrackingRequest:
    direction: str
    start_frame: int
    end_frame: int
    seeds: List[TrackingSeed]
    engine: str = "deterministic"
    replace_manual: bool = False

    def validate(self) -> None:
        if self.direction not in {"forward", "backward", "bidirectional"}:
            raise ValueError("Invalid tracking direction: {}".format(self.direction))
        self.start_frame = max(0, int(self.start_frame))
        self.end_frame = max(0, int(self.end_frame))
        if self.end_frame < self.start_frame:
            raise ValueError("Tracking end frame cannot precede start frame")
        if not self.seeds:
            raise ValueError("At least one reviewed correction/key frame is required")
        for seed in self.seeds:
            seed.frame_index = int(seed.frame_index)
            seed.object_id = str(seed.object_id)
            seed.mask_path = Path(seed.mask_path)
            if not seed.object_id or not seed.mask_path.is_file():
                raise ValueError("Tracking seed masks must exist and have stable object IDs")
            if not self.start_frame <= seed.frame_index <= self.end_frame:
                raise ValueError("Tracking seed is outside the selected range")


class TrackingEngineAdapter(Protocol):
    name: str

    def propagate(
        self,
        request: TrackingRequest,
        frame_paths: Mapping[int, Path],
        cancel_event: Event,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Iterable[Tuple[int, Mapping[str, np.ndarray]]]:
        ...


class DeterministicTrackingAdapter:
    """Test/reference adapter that copies the nearest correction mask."""

    name = "deterministic"

    def propagate(
        self,
        request: TrackingRequest,
        frame_paths: Mapping[int, Path],
        cancel_event: Event,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Iterable[Tuple[int, Mapping[str, np.ndarray]]]:
        del frame_paths
        by_object: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        for seed in request.seeds:
            mask = cv2.imread(str(seed.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Unable to read seed mask: {}".format(seed.mask_path))
            by_object.setdefault(seed.object_id, []).append((seed.frame_index, mask > 0))
        indices = list(range(request.start_frame, request.end_frame + 1))
        if request.direction == "backward":
            indices.reverse()
        total = len(indices)
        for position, frame_index in enumerate(indices, start=1):
            if cancel_event.is_set():
                raise OperationCancelled("Tracking was cancelled")
            masks = {}
            for object_id, corrections in by_object.items():
                _seed_frame, mask = min(
                    corrections, key=lambda item: abs(item[0] - frame_index)
                )
                masks[object_id] = mask.copy()
            if progress is not None:
                progress(position, total)
            yield frame_index, masks


class SegAnyTrackingAdapter:
    """Adapter around the existing SAM2/SAM3/SAM3.1 video facade.

    The adapter owns no widgets and never writes accepted labels. It receives a
    model facade that has already been loaded by the application.
    """

    name = "segany-video"

    def __init__(self, facade: Any) -> None:
        self.facade = facade

    @staticmethod
    def _mask_array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        value = np.asarray(value)
        while value.ndim > 2:
            value = value[0]
        return value > 0

    def _propagate_direction(
        self,
        request: TrackingRequest,
        reverse: bool,
        cancel_event: Event,
    ) -> Iterator[Tuple[int, Mapping[str, np.ndarray]]]:
        seed_frame = (
            max(seed.frame_index for seed in request.seeds)
            if reverse
            else min(seed.frame_index for seed in request.seeds)
        )
        count = (
            seed_frame - request.start_frame + 1
            if reverse
            else request.end_frame - seed_frame + 1
        )
        iterator = self.facade.predictor.propagate_in_video(
            self.facade.inference_state,
            start_frame_idx=seed_frame,
            max_frame_num_to_track=count,
            reverse=reverse,
        )
        instance_by_model_id = {
            index: object_id
            for index, object_id in enumerate(
                dict.fromkeys(seed.object_id for seed in request.seeds)
            )
        }
        for frame_index, object_ids, logits in iterator:
            if cancel_event.is_set():
                raise OperationCancelled("Tracking was cancelled")
            index = int(frame_index)
            if not request.start_frame <= index <= request.end_frame:
                continue
            masks: Dict[str, np.ndarray] = {}
            for mask_index, model_object_id in enumerate(object_ids):
                raw_id = int(model_object_id)
                object_id = instance_by_model_id.get(raw_id, str(raw_id))
                masks[object_id] = self._mask_array(logits[mask_index])
            yield index, masks

    def propagate(
        self,
        request: TrackingRequest,
        frame_paths: Mapping[int, Path],
        cancel_event: Event,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Iterable[Tuple[int, Mapping[str, np.ndarray]]]:
        ordered_indices = sorted(frame_paths)
        if ordered_indices != list(range(request.start_frame, request.end_frame + 1)):
            raise ValueError("Model tracking requires a dense decoded frame range")
        root = Path(frame_paths[ordered_indices[0]]).parent
        names = [Path(frame_paths[index]).name for index in ordered_indices]
        self.facade.init_state(str(root), names, async_loading_frames=True)
        self.facade.reset_state()
        model_ids: Dict[str, int] = {
            object_id: index
            for index, object_id in enumerate(
                dict.fromkeys(seed.object_id for seed in request.seeds)
            )
        }
        for seed in request.seeds:
            mask = cv2.imread(str(seed.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("Unable to read seed mask: {}".format(seed.mask_path))
            self.facade.add_new_mask(
                seed.frame_index - request.start_frame,
                model_ids[seed.object_id],
                mask > 0,
            )

        directions = [False]
        if request.direction == "backward":
            directions = [True]
        elif request.direction == "bidirectional":
            directions = [True, False]
        emitted = set()
        expected = request.end_frame - request.start_frame + 1
        completed = 0
        for reverse in directions:
            for relative_index, masks in self._propagate_direction(
                TrackingRequest(
                    direction=request.direction,
                    start_frame=0,
                    end_frame=request.end_frame - request.start_frame,
                    seeds=[
                        TrackingSeed(
                            frame_index=seed.frame_index - request.start_frame,
                            object_id=seed.object_id,
                            mask_path=seed.mask_path,
                        )
                        for seed in request.seeds
                    ],
                    engine=request.engine,
                ),
                reverse,
                cancel_event,
            ):
                absolute_index = relative_index + request.start_frame
                if absolute_index in emitted:
                    continue
                emitted.add(absolute_index)
                completed += 1
                if progress is not None:
                    progress(completed, expected)
                yield absolute_index, masks


def _stored_path(store: UnifiedVideoProjectStore, path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(store.case_directory.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_path(store: UnifiedVideoProjectStore, value: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = store.case_directory / path
    return path.resolve()


class VideoTrackingService:
    def __init__(self, store: UnifiedVideoProjectStore) -> None:
        self.store = store

    def stage(
        self,
        request: TrackingRequest,
        adapter: TrackingEngineAdapter,
        frame_paths: Mapping[int, Path],
        cancel_event: Optional[Event] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        request.validate()
        cancellation = cancel_event or Event()
        known_instances = {item["object_id"] for item in self.store.project["instances"]}
        requested_instances = list(dict.fromkeys(seed.object_id for seed in request.seeds))
        if any(item not in known_instances for item in requested_instances):
            raise ValueError("Tracking seeds must reference registered project instances")
        with self.store.transaction():
            run = self.store.create_tracking_run(
                request.direction,
                request.start_frame,
                request.end_frame,
                requested_instances,
                getattr(adapter, "name", request.engine),
                replace_manual=request.replace_manual,
            )
            self.store.save()
        run_directory = self.store.tracking_directory / run["id"]
        masks_directory = run_directory / "staged_masks"
        masks_directory.mkdir(parents=True, exist_ok=True)
        self.store.update_tracking_run(run["id"], status="running")
        self.store.save()
        staged_by_frame: Dict[int, Dict[str, Any]] = {}
        try:
            for frame_index, masks in adapter.propagate(
                request, frame_paths, cancellation, progress
            ):
                if cancellation.is_set():
                    raise OperationCancelled("Tracking was cancelled")
                frame_entry = staged_by_frame.setdefault(
                    int(frame_index), {"frame_index": int(frame_index), "objects": []}
                )
                for object_id, mask in masks.items():
                    mask_path = masks_directory / "frame_{:012d}_{}.png".format(
                        int(frame_index), str(object_id)
                    )
                    atomic_write_png(mask_path, np.asarray(mask, dtype=np.uint8) * 255)
                    frame_entry["objects"].append(
                        {
                            "object_id": str(object_id),
                            "mask_path": _stored_path(self.store, mask_path),
                            "review_status": "draft",
                        }
                    )
            staged_frames = [staged_by_frame[index] for index in sorted(staged_by_frame)]
            if not staged_frames:
                raise ValueError("Tracking engine produced no staged frames")
            with self.store.transaction():
                run = self.store.update_tracking_run(
                    run["id"],
                    status="staged",
                    staged_frames=staged_frames,
                    metrics={
                        "frame_count": len(staged_frames),
                        "object_count": len(requested_instances),
                    },
                )
                self.store.save()
            return run
        except OperationCancelled as error:
            with self.store.transaction():
                self.store.update_tracking_run(
                    run["id"],
                    status="cancelled",
                    staged_frames=[staged_by_frame[index] for index in sorted(staged_by_frame)],
                    error=str(error),
                )
                self.store.save()
            raise
        except Exception as error:
            with self.store.transaction():
                self.store.update_tracking_run(
                    run["id"],
                    status="failed",
                    staged_frames=[staged_by_frame[index] for index in sorted(staged_by_frame)],
                    error=str(error),
                )
                self.store.save()
            raise

    @staticmethod
    def _mask_objects(
        mask: np.ndarray,
        instance: Mapping[str, Any],
        group: int,
    ) -> List[Object]:
        contours, _hierarchy = cv2.findContours(
            np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        objects = []
        for contour in contours:
            if len(contour) < 3:
                continue
            segmentation = [
                (float(point[0][0]), float(point[0][1])) for point in contour
            ]
            area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            objects.append(
                Object(
                    category=str(instance.get("category") or "unknown"),
                    group=group,
                    segmentation=segmentation,
                    area=area,
                    layer=2,
                    bbox=(x, y, x + width, y + height),
                    note="Tracked from reviewed key frame",
                    object_id=str(instance["object_id"]),
                    track_id=str(instance["track_id"]),
                    source="tracking",
                    review_status="draft",
                )
            )
        return objects

    def _run(self, run_id: str) -> Dict[str, Any]:
        run = next(
            (raw for raw in self.store.project["tracking_runs"] if raw.get("id") == run_id),
            None,
        )
        if run is None:
            raise KeyError("Unknown tracking run: {}".format(run_id))
        return run

    def commit(
        self,
        run_id: str,
        image_path_for_frame: Callable[[int], Path],
        label_path_for_frame: Callable[[int], Path],
        replace_manual: bool = False,
    ) -> Dict[str, Any]:
        run = self._run(run_id)
        if run.get("status") != "staged":
            raise ValueError("Only staged tracking runs can be committed")
        instances = {item["object_id"]: item for item in self.store.project["instances"]}
        backup_directory = self.store.tracking_directory / run_id / "accepted_backup"
        backup_directory.mkdir(parents=True, exist_ok=True)
        backup_manifest: List[Dict[str, Any]] = []
        modified: List[Path] = []
        store_snapshot = self.store._runtime_snapshot()
        frame_map = None
        frame_map_info = self.store.project.get("frame_map", {})
        if frame_map_info.get("asset_path"):
            try:
                frame_map = FrameTimestampMap.load(
                    _resolve_path(self.store, frame_map_info["asset_path"])
                )
            except Exception:
                frame_map = None
        try:
            for frame_entry in run.get("staged_frames", []):
                frame_index = int(frame_entry["frame_index"])
                image_path = Path(image_path_for_frame(frame_index)).resolve()
                label_path = Path(label_path_for_frame(frame_index)).resolve()
                annotation = Annotation(str(image_path), str(label_path)).load_annotation()
                backup_path = backup_directory / "frame_{:012d}.json".format(frame_index)
                existed = label_path.is_file()
                if existed:
                    shutil.copy2(label_path, backup_path)
                backup_manifest.append(
                    {
                        "frame_index": frame_index,
                        "label_path": _stored_path(self.store, label_path),
                        "backup_path": _stored_path(self.store, backup_path) if existed else "",
                        "existed": existed,
                    }
                )
                incoming_track_ids = {
                    str(instances[item["object_id"]]["track_id"])
                    for item in frame_entry.get("objects", [])
                    if item.get("object_id") in instances
                }
                manual_conflicts = {
                    obj.track_id
                    for obj in annotation.objects
                    if obj.track_id in incoming_track_ids and obj.source != "tracking"
                }
                annotation.objects = [
                    obj
                    for obj in annotation.objects
                    if not (
                        obj.track_id in incoming_track_ids
                        and (obj.source == "tracking" or replace_manual)
                    )
                ]
                next_group = max([int(obj.group) for obj in annotation.objects] + [0]) + 1
                for item in frame_entry.get("objects", []):
                    instance = instances.get(str(item.get("object_id")))
                    if instance is None:
                        continue
                    if instance["track_id"] in manual_conflicts and not replace_manual:
                        continue
                    mask = cv2.imread(
                        str(_resolve_path(self.store, item["mask_path"])),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    if mask is None:
                        raise ValueError("Missing staged tracking mask")
                    annotation.objects.extend(
                        self._mask_objects(mask > 0, instance, next_group)
                    )
                    next_group += 1
                annotation.save_annotation()
                modified.append(label_path)
                fps = float(self.store.media.get("fps", 0.0) or 0.0)
                pts_ms = (
                    frame_map.timestamp_for_frame(frame_index)
                    if frame_map is not None
                    else int(round(frame_index * 1000.0 / fps)) if fps > 0 else 0
                )
                self.store.add_spatial_frame(
                    frame_index,
                    pts_ms,
                    image_path,
                    label_path,
                    annotation_status="saved",
                )

            manifest_path = backup_directory / "backup_manifest.json"
            manifest_path.write_text(
                json.dumps(backup_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with self.store.transaction():
                updated = self.store.update_tracking_run(
                    run_id, status="committed", committed_at=utc_now()
                )
                self.store.save()
            return updated
        except Exception:
            self.store._restore_runtime_snapshot(store_snapshot)
            for entry in reversed(backup_manifest):
                label_path = _resolve_path(self.store, entry["label_path"])
                backup_path = (
                    _resolve_path(self.store, entry["backup_path"])
                    if entry.get("backup_path")
                    else None
                )
                if entry.get("existed") and backup_path is not None and backup_path.is_file():
                    shutil.copy2(backup_path, label_path)
                elif not entry.get("existed") and label_path.is_file():
                    label_path.unlink()
            raise

    def rollback(self, run_id: str) -> Dict[str, Any]:
        run = self._run(run_id)
        if run.get("status") != "committed":
            raise ValueError("Only committed tracking runs can be rolled back")
        manifest_path = (
            self.store.tracking_directory
            / run_id
            / "accepted_backup"
            / "backup_manifest.json"
        )
        if not manifest_path.is_file():
            raise ValueError("Tracking rollback manifest is missing")
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in entries:
            label_path = _resolve_path(self.store, entry["label_path"])
            if entry.get("existed"):
                backup_path = _resolve_path(self.store, entry["backup_path"])
                if not backup_path.is_file():
                    raise ValueError("Tracking rollback backup is missing")
                shutil.copy2(backup_path, label_path)
            elif label_path.is_file():
                label_path.unlink()
        with self.store.transaction():
            updated = self.store.update_tracking_run(
                run_id, status="rolled_back", rolled_back_at=utc_now()
            )
            self.store.save()
        return updated
