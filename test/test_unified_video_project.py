import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from IVAS.video.project import (
    SCHEMA_VERSION,
    UnifiedVideoProjectStore,
    case_directory,
)


class UnifiedVideoProjectTests(unittest.TestCase):
    def _video(self, root: Path) -> Path:
        path = root / "case.mp4"
        path.write_bytes(b"synthetic-video-payload")
        return path

    def test_interval_and_spatial_frame_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            project_path = case_directory(root / "outputs", video) / "project.ivas.json"
            store = UnifiedVideoProjectStore.create(
                project_path,
                video,
                duration_ms=10_000,
                fps=25.0,
                frame_count=250,
                width=640,
                height=480,
            )
            interval = store.add_interval("phase", "Incision", 1_000, 2_500)
            frame_path = store.frames_directory / "frame_00000025.png"
            label_path = store.labels_directory / "frame_00000025.json"
            store.add_spatial_frame(25, 1_000, frame_path, label_path)
            store.save()

            loaded = UnifiedVideoProjectStore.load(project_path)

            self.assertEqual(loaded.project["schema_version"], SCHEMA_VERSION)
            self.assertEqual(loaded.temporal_annotations[0].id, interval.id)
            self.assertEqual(loaded.temporal_annotations[0].label, "Incision")
            self.assertEqual(loaded.temporal_annotations[0].status, "draft")
            self.assertEqual(loaded.project["spatial_frames"][0]["frame_index"], 25)
            self.assertEqual(
                loaded.project["spatial_frames"][0]["annotation_status"], "pending"
            )
            self.assertEqual(
                loaded.project["spatial_frames"][0]["image_path"],
                "frames/frame_00000025.png",
            )

    def test_invalid_interval_does_not_mutate_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            store = UnifiedVideoProjectStore.create(
                root / "project.ivas.json", video, duration_ms=5_000
            )

            with self.assertRaises(ValueError):
                store.add_interval("phase", "Invalid", 4_000, 3_000)

            self.assertEqual(store.temporal_annotations, [])

    def test_undo_and_redo_cover_temporal_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            store = UnifiedVideoProjectStore.create(root / "project.ivas.json", video)
            store.add_interval("task", "Irrigation", 10, 20)
            audit_before_undo = list(store.project["audit"])

            self.assertTrue(store.undo())
            self.assertEqual(len(store.temporal_annotations), 0)
            self.assertEqual(store.project["audit"][:-1], audit_before_undo)
            self.assertEqual(store.project["audit"][-1]["action"], "undo")
            self.assertTrue(store.redo())
            self.assertEqual(store.temporal_annotations[0].label, "Irrigation")
            self.assertEqual(store.project["audit"][:-1][-1]["action"], "undo")
            self.assertEqual(store.project["audit"][-1]["action"], "redo")

    def test_open_unchanged_project_is_clean_and_clean_save_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            output_root = root / "outputs"
            project_path = case_directory(output_root, video) / "project.ivas.json"
            store = UnifiedVideoProjectStore.create(
                project_path,
                video,
                duration_ms=10_000,
                fps=29.97002997003,
                frame_count=300,
                width=640,
                height=480,
            )
            store.save()
            store.add_interval("phase", "Incision", 1_000, 2_000)
            store.save()
            backup_path = project_path.with_name(
                "{}.previous{}".format(project_path.stem, project_path.suffix)
            )
            backup_before = backup_path.read_bytes()

            opened = UnifiedVideoProjectStore.open_or_create(
                output_root,
                video,
                duration_ms=10_000,
                fps=29.97002997003,
                frame_count=300,
                width=640,
                height=480,
            )

            self.assertFalse(opened.is_dirty)
            with mock.patch(
                "IVAS.video.project.os.replace",
                side_effect=AssertionError("clean save must not replace files"),
            ):
                opened.save()
            self.assertEqual(backup_path.read_bytes(), backup_before)

    def test_media_metadata_update_is_outside_video_annotation_undo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            store = UnifiedVideoProjectStore.create(
                root / "project.ivas.json", video, duration_ms=1_000, fps=25.0
            )
            store.save()

            store.update_media_info(duration_ms=2_000, fps=30.0)

            self.assertTrue(store.is_dirty)
            self.assertEqual(store.media["duration_ms"], 2_000)
            self.assertEqual(store.media["fps"], 30.0)
            self.assertFalse(store.undo())
            self.assertEqual(store.media["duration_ms"], 2_000)
            self.assertEqual(store.media["fps"], 30.0)

    def test_mutator_exception_rolls_back_project_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            store = UnifiedVideoProjectStore.create(root / "project.ivas.json", video)
            store.save()
            project_before = store._snapshot()

            with mock.patch.object(
                store, "_changed", side_effect=RuntimeError("audit failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "audit failed"):
                    store.add_interval("phase", "Incision", 10, 20)

            self.assertEqual(store.project, project_before)
            self.assertFalse(store.is_dirty)
            self.assertFalse(store.can_undo)
            self.assertFalse(store.can_redo)

    def test_transaction_rolls_back_mutation_when_save_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            project_path = root / "project.ivas.json"
            store = UnifiedVideoProjectStore.create(project_path, video)
            store.save()
            project_before = store._snapshot()
            disk_before = project_path.read_bytes()

            with mock.patch(
                "IVAS.video.project.os.replace", side_effect=OSError("disk full")
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    with store.transaction():
                        store.add_interval("phase", "Incision", 10, 20)
                        store.save()

            self.assertEqual(store.project, project_before)
            self.assertEqual(project_path.read_bytes(), disk_before)
            self.assertFalse(store.is_dirty)
            self.assertFalse(store.can_undo)
            self.assertFalse(store.can_redo)

    def test_load_recovers_from_previous_atomic_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            project_path = root / "project.ivas.json"
            store = UnifiedVideoProjectStore.create(project_path, video)
            store.save()
            store.add_interval("phase", "Second revision", 10, 20)
            store.save()
            project_path.write_text("{broken", encoding="utf-8")

            recovered = UnifiedVideoProjectStore.load(project_path)

            self.assertTrue(recovered.recovered_from_backup)
            self.assertEqual(recovered.temporal_annotations, [])
            recovered.save()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)

    def test_open_or_create_migrates_legacy_surgui_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            case_root = case_directory(root / "outputs", video)
            case_root.mkdir(parents=True)
            (case_root / "project.surgui.json").write_text(
                json.dumps(
                    {
                        "project_id": "legacy-case-1",
                        "created_at": "2025-01-02T03:04:05+00:00",
                        "media": {"path": str(video)},
                        "audit": [{"action": "legacy_edit", "timestamp": "then"}],
                        "templates": [
                            {
                                "id": "assessment_1",
                                "label": "Safety",
                                "kind": "rating",
                                "category": "assessment",
                            }
                        ],
                        "annotations": [
                            {
                                "id": "legacy_rating",
                                "kind": "rating",
                                "template_id": "assessment_1",
                                "label": "Safety",
                                "category": "assessment",
                                "start_ms": 0,
                                "score": 4,
                                "status": "reviewed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            store = UnifiedVideoProjectStore.open_or_create(
                root / "outputs", video, duration_ms=1_000, fps=25.0
            )

            self.assertEqual(store.temporal_annotations[0].id, "legacy_rating")
            self.assertEqual(store.temporal_annotations[0].score, 4)
            self.assertEqual(
                store.project["temporal_templates"][0]["id"], "assessment_1"
            )
            self.assertTrue(store.project["legacy_source"].endswith("project.surgui.json"))
            self.assertIn("legacy_project", store.project)
            self.assertEqual(
                store.project["legacy_project"]["project_id"], "legacy-case-1"
            )
            self.assertEqual(
                store.project["legacy_project"]["audit"][0]["action"],
                "legacy_edit",
            )


if __name__ == "__main__":
    unittest.main()
