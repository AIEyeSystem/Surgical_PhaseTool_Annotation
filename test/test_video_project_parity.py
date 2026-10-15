import csv
import json
import tempfile
import unittest
from pathlib import Path

from IVAS.video.project import (
    MediaRelinkRequired,
    SCHEMA_VERSION,
    TemporalTemplate,
    UnifiedVideoProjectStore,
    discover_video_projects,
)


class VideoProjectParityTests(unittest.TestCase):
    def _store(self, root: Path) -> UnifiedVideoProjectStore:
        root.mkdir(parents=True, exist_ok=True)
        media = root / "case.mp4"
        media.write_bytes(b"video-a")
        store = UnifiedVideoProjectStore.create(
            root / "case" / "project.ivas.json",
            media,
            duration_ms=10_000,
            fps=25,
            frame_count=250,
        )
        store.save()
        return store

    def test_schema_two_migrates_without_losing_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            payload = json.loads(store.path.read_text(encoding="utf-8"))
            payload["schema_version"] = 2
            payload.pop("frame_map", None)
            payload.pop("media_history", None)
            payload["temporal_annotations"] = [
                {
                    "id": "legacy",
                    "kind": "interval",
                    "category": "phase",
                    "label": "Incision",
                    "start_ms": 100,
                    "end_ms": 900,
                }
            ]
            store.path.write_text(json.dumps(payload), encoding="utf-8")

            migrated = UnifiedVideoProjectStore.load(store.path)
            self.assertEqual(migrated.project["schema_version"], SCHEMA_VERSION)
            self.assertEqual(migrated.temporal_annotations[0].label, "Incision")
            self.assertTrue(migrated.temporal_annotations[0].template_id)
            self.assertTrue(migrated.is_dirty)

    def test_arbitrary_project_open_requires_explicit_identity_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            original = Path(store.media["path"])
            moved = root / "moved.mp4"
            moved.write_bytes(original.read_bytes())
            moved.touch()

            with self.assertRaises(MediaRelinkRequired):
                UnifiedVideoProjectStore.open_existing(store.path, moved)
            rebound = UnifiedVideoProjectStore.open_existing(
                store.path, moved, allow_identity_change=True
            )
            self.assertEqual(Path(rebound.media["path"]), moved.resolve())
            self.assertEqual(len(rebound.project["media_history"]), 1)
            rebound.save()
            self.assertIn(store.path.resolve(), discover_video_projects(root))

    def test_template_temporal_crud_validation_and_interchange(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            phase = store.add_template(
                TemporalTemplate(
                    id="phase-incision",
                    label="Incision",
                    kind="interval",
                    category="phase",
                    required=True,
                )
            )
            point = store.add_template(
                TemporalTemplate(
                    id="key-review",
                    label="Review frame",
                    kind="point",
                    category="keyframe",
                )
            )
            rating = store.add_template(
                TemporalTemplate(
                    id="quality",
                    label="Quality",
                    kind="rating",
                    category="assessment",
                    score_options=[0, 1, 2],
                    required=True,
                )
            )
            interval = store.add_temporal_annotation(
                phase.id, start_ms=0, end_ms=5_000, note="first", status="ready"
            )
            keyframe = store.add_point(
                point.id, start_ms=2_000, frame_index=50, note="correction"
            )
            score = store.set_rating(rating.id, 2, status="reviewed")
            updated = store.update_temporal_annotation(
                keyframe.id, note="reviewed correction", status="reviewed"
            )
            self.assertEqual(updated.note, "reviewed correction")
            self.assertEqual(score.score, 2)
            self.assertAlmostEqual(store.phase_coverage_ratio(), 0.5)
            issues = store.validate()
            self.assertFalse(any(issue.severity == "error" for issue in issues))

            csv_path = store.export_csv(root / "export" / "annotations.csv")
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            legacy_paths = store.export_legacy(root / "legacy")
            self.assertTrue(legacy_paths)

            default_csv, default_legacy = store.sync_default_exports()
            self.assertEqual(default_csv, store.case_directory / "exports" / "annotations.csv")
            self.assertTrue(default_csv.is_file())
            self.assertTrue(default_legacy)
            phase_export = store.case_directory / "exports" / "phase.txt"
            self.assertTrue(phase_export.is_file())

            clone = self._store(root / "clone")
            clone.append_templates([phase, point, rating])
            reports = clone.import_csv(csv_path)
            self.assertFalse(any(issue.severity == "error" for issue in reports))
            self.assertEqual(len(clone.temporal_annotations), 3)
            clone.save()
            reopened = UnifiedVideoProjectStore.load(clone.path)
            self.assertEqual(len(reopened.temporal_annotations), 3)

            deleted = store.delete_temporal_annotation(interval.id)
            self.assertEqual(deleted.id, interval.id)
            store.sync_default_exports()
            self.assertFalse(phase_export.exists())
            store.delete_template(phase.id)


if __name__ == "__main__":
    unittest.main()
