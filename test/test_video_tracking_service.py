import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest import mock

import cv2
import numpy as np

from IVAS.annotation import Annotation, Object
from IVAS.video.project import UnifiedVideoProjectStore
from IVAS.video.tracking import (
    DeterministicTrackingAdapter,
    TrackingRequest,
    TrackingSeed,
    VideoTrackingService,
)
from IVAS.video.media import OperationCancelled


class VideoTrackingServiceTests(unittest.TestCase):
    def test_stage_manual_wins_commit_reopen_and_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "case.mp4"
            media.write_bytes(b"video")
            store = UnifiedVideoProjectStore.create(
                root / "case" / "project.ivas.json",
                media,
                duration_ms=120,
                fps=25,
                frame_count=3,
            )
            instance = store.ensure_instance(
                "instrument", "Forceps", object_id="forceps", track_id="track-forceps"
            )
            store.save()

            frame_paths = {}
            for index in range(3):
                path = store.frames_directory / "frame_{:012d}.png".format(index)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), np.zeros((16, 16, 3), dtype=np.uint8))
                frame_paths[index] = path
            seed_path = root / "seed.png"
            seed = np.zeros((16, 16), dtype=np.uint8)
            seed[3:10, 4:12] = 255
            cv2.imwrite(str(seed_path), seed)

            service = VideoTrackingService(store)
            run = service.stage(
                TrackingRequest(
                    direction="bidirectional",
                    start_frame=0,
                    end_frame=2,
                    seeds=[TrackingSeed(1, instance["object_id"], seed_path)],
                ),
                DeterministicTrackingAdapter(),
                frame_paths,
                Event(),
            )
            self.assertEqual(run["status"], "staged")
            self.assertEqual(len(run["staged_frames"]), 3)

            manual_label = store.labels_directory / "frame_000000000001.json"
            manual = Annotation(str(frame_paths[1]), str(manual_label))
            manual.objects = [
                Object(
                    "instrument",
                    1,
                    [(1, 1), (6, 1), (6, 6)],
                    12.5,
                    2,
                    (1, 1, 6, 6),
                    object_id="manual-forceps",
                    track_id="track-forceps",
                    source="manual",
                )
            ]
            manual.save_annotation()

            def image_path(index):
                return frame_paths[index]

            def label_path(index):
                return store.labels_directory / "frame_{:012d}.json".format(index)

            committed = service.commit(run["id"], image_path, label_path)
            self.assertEqual(committed["status"], "committed")
            preserved = Annotation(str(frame_paths[1]), str(manual_label)).load_annotation()
            self.assertEqual(len(preserved.objects), 1)
            self.assertEqual(preserved.objects[0].source, "manual")
            tracked = Annotation(str(frame_paths[0]), str(label_path(0))).load_annotation()
            self.assertTrue(tracked.objects)
            self.assertTrue(all(item.source == "tracking" for item in tracked.objects))

            reopened = UnifiedVideoProjectStore.load(store.path)
            reopened_run = next(item for item in reopened.project["tracking_runs"] if item["id"] == run["id"])
            self.assertEqual(reopened_run["status"], "committed")
            rolled_back = VideoTrackingService(reopened).rollback(run["id"])
            self.assertEqual(rolled_back["status"], "rolled_back")
            self.assertFalse(label_path(0).exists())
            restored = Annotation(str(frame_paths[1]), str(manual_label)).load_annotation()
            self.assertEqual(restored.objects[0].source, "manual")

    def test_cancelled_stage_is_recorded_without_accepted_labels(self):
        class CancellingAdapter:
            name = "cancel-test"

            def propagate(self, request, frame_paths, cancel_event, progress=None):
                mask = np.ones((8, 8), dtype=np.uint8)
                yield request.start_frame, {request.seeds[0].object_id: mask}
                cancel_event.set()
                yield request.end_frame, {request.seeds[0].object_id: mask}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "case.mp4"
            media.write_bytes(b"video")
            store = UnifiedVideoProjectStore.create(
                root / "case" / "project.ivas.json", media, 120, 25, 2
            )
            instance = store.ensure_instance("instrument", "Forceps")
            seed_path = root / "seed.png"
            cv2.imwrite(str(seed_path), np.ones((8, 8), dtype=np.uint8) * 255)
            frames = {}
            for index in range(2):
                frame = store.frames_directory / "frame_{:012d}.png".format(index)
                frame.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame), np.zeros((8, 8, 3), dtype=np.uint8))
                frames[index] = frame

            with self.assertRaises(OperationCancelled):
                VideoTrackingService(store).stage(
                    TrackingRequest(
                        "forward",
                        0,
                        1,
                        [TrackingSeed(0, instance["object_id"], seed_path)],
                    ),
                    CancellingAdapter(),
                    frames,
                    Event(),
                )
            self.assertEqual(store.project["tracking_runs"][-1]["status"], "cancelled")
            self.assertEqual(list(store.labels_directory.glob("*.json")), [])

    def test_commit_failure_restores_every_label_and_staged_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "case.mp4"
            media.write_bytes(b"video")
            store = UnifiedVideoProjectStore.create(
                root / "case" / "project.ivas.json", media, 120, 25, 2
            )
            instance = store.ensure_instance("instrument", "Forceps")
            frames = {}
            for index in range(2):
                frame = store.frames_directory / "frame_{:012d}.png".format(index)
                frame.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame), np.zeros((8, 8, 3), dtype=np.uint8))
                frames[index] = frame
            seed_path = root / "seed.png"
            cv2.imwrite(str(seed_path), np.ones((8, 8), dtype=np.uint8) * 255)
            service = VideoTrackingService(store)
            run = service.stage(
                TrackingRequest(
                    "forward", 0, 1, [TrackingSeed(0, instance["object_id"], seed_path)]
                ),
                DeterministicTrackingAdapter(),
                frames,
                Event(),
            )
            existing_label = store.labels_directory / "frame_000000000001.json"
            existing = Annotation(str(frames[1]), str(existing_label))
            existing.objects = [
                Object(
                    "manual-category",
                    1,
                    [(1, 1), (5, 1), (5, 5)],
                    8.0,
                    2,
                    (1, 1, 5, 5),
                    source="manual",
                )
            ]
            existing.save_annotation()
            before = existing_label.read_bytes()
            original_save = Annotation.save_annotation
            calls = 0

            def fail_second(annotation):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated disk failure")
                return original_save(annotation)

            with mock.patch.object(Annotation, "save_annotation", fail_second):
                with self.assertRaises(OSError):
                    service.commit(
                        run["id"],
                        lambda index: frames[index],
                        lambda index: store.labels_directory
                        / "frame_{:012d}.json".format(index),
                    )
            self.assertFalse((store.labels_directory / "frame_000000000000.json").exists())
            self.assertEqual(existing_label.read_bytes(), before)
            self.assertEqual(service._run(run["id"])["status"], "staged")


if __name__ == "__main__":
    unittest.main()
