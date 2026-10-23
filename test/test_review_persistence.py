"""Concurrent writers and interrupted multi-file tracking transactions."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from IVAS.annotation import Annotation, Object
from IVAS.video.merge import ProjectConflictError
from IVAS.video.project import UnifiedVideoProjectStore
from IVAS.video.tracking import (DeterministicTrackingAdapter, TrackingRequest,
                                 TrackingSeed, VideoTrackingService)


class PersistenceReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        media = self.root / "video.mp4"
        media.write_bytes(b"video")
        self.store = UnifiedVideoProjectStore.create(self.root / "case/project.ivas.json", media, 400, 10, 4)
        self.store.ensure_instance("tool", "Tool", object_id="object-1", track_id="track-1")
        self.store.save()
        self.frames = {}
        for index in range(2):
            path = self.store.frames_directory / ("frame_%012d.png" % index)
            Image.new("RGB", (16, 16)).save(path)
            self.frames[index] = path
        seed = self.root / "seed.png"
        mask = np.zeros((16, 16), np.uint8)
        mask[2:8, 2:8] = 255
        Image.fromarray(mask).save(seed)
        self.request = TrackingRequest("forward", 0, 1, [TrackingSeed(0, "object-1", seed)])
        self.service = VideoTrackingService(self.store)

    def label(self, index):
        return self.store.labels_directory / ("frame_%012d.json" % index)

    def stage(self):
        return self.service.stage(self.request, DeterministicTrackingAdapter(), self.frames, threading.Event())

    def commit(self, run):
        return self.service.commit(run["id"], self.frames.__getitem__, self.label)

    def test_temporal_edit_during_tracking_survives_and_keeps_undo(self):
        worker = UnifiedVideoProjectStore.load(self.store.path)
        ui = self.store
        class Interleaved(DeterministicTrackingAdapter):
            def propagate(adapter, *args, **kwargs):
                ui.add_interval("phase", "during tracking", 0, 100)
                ui.save()
                yield from super().propagate(*args, **kwargs)
        VideoTrackingService(worker).stage(self.request, Interleaved(), self.frames, threading.Event())
        ui.refresh()
        self.assertEqual(len(ui.temporal_annotations), 1)
        self.assertEqual(ui.project["tracking_runs"][0]["status"], "staged")
        self.assertTrue(ui.undo())
        ui.save()
        disk = UnifiedVideoProjectStore.load(ui.path)
        self.assertEqual(len(disk.temporal_annotations), 0)
        self.assertEqual(disk.project["tracking_runs"][0]["status"], "staged")

    def test_simultaneous_independent_saves_merge(self):
        writers = [UnifiedVideoProjectStore.load(self.store.path) for _ in range(3)]
        barrier = threading.Barrier(len(writers))
        errors = []
        def save(index):
            try:
                writers[index].add_interval("phase", str(index), index * 100, index * 100 + 90)
                barrier.wait(timeout=5)
                writers[index].save()
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=save, args=(i,)) for i in range(len(writers))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(UnifiedVideoProjectStore.load(self.store.path).temporal_annotations), 3)

    def test_conflicting_edit_does_not_overwrite_disk_or_local_state(self):
        other = UnifiedVideoProjectStore.load(self.store.path)
        self.store.project["instances"][0]["display_name"] = "first"
        other.project["instances"][0]["display_name"] = "second"
        self.store.is_dirty = other.is_dirty = True
        self.store.save()
        before_disk, before_local = self.store.path.read_bytes(), copy.deepcopy(other.project)
        with self.assertRaises(ProjectConflictError):
            other.save()
        self.assertEqual(other.project, before_local)
        self.assertEqual(self.store.path.read_bytes(), before_disk)

    def test_replaced_project_identity_is_rejected(self):
        payload = json.loads(self.store.path.read_text(encoding="utf-8"))
        payload["project_id"] = "another-project"
        self.store.path.write_text(json.dumps(payload), encoding="utf-8")
        self.store.is_dirty = True
        with self.assertRaises(ProjectConflictError):
            self.store.save()

    def test_failed_manifest_rollback_restores_all_labels_and_runtime(self):
        run = self.stage()
        self.commit(run)
        before = [self.label(i).read_bytes() for i in range(2)]
        project = copy.deepcopy(self.store.project)
        with mock.patch.object(self.store, "save", side_effect=OSError("injected failure")):
            with self.assertRaises(OSError):
                self.service.rollback(run["id"])
        self.assertEqual([self.label(i).read_bytes() for i in range(2)], before)
        self.assertEqual(self.store.project, project)
        self.assertEqual(UnifiedVideoProjectStore.load(self.store.path).project, project)

    def test_successful_rollback_restores_spatial_metadata(self):
        run = self.stage()
        spatial = copy.deepcopy(self.store.project["spatial_frames"])
        self.commit(run)
        self.service.rollback(run["id"])
        self.assertEqual(self.store.project["spatial_frames"], spatial)
        self.assertFalse(self.label(0).exists())
        self.assertFalse(self.label(1).exists())

    def test_error_after_manifest_replace_keeps_committed_labels(self):
        run = self.stage()
        save = self.store.save
        def save_then_fail():
            save()
            raise OSError("failure after durable commit")
        with mock.patch.object(self.store, "save", side_effect=save_then_fail):
            with self.assertRaises(OSError):
                self.commit(run)
        self.assertTrue(self.label(0).is_file())
        self.assertTrue(self.label(1).is_file())
        self.assertEqual(self.service._run(run["id"])["status"], "committed")
        disk = UnifiedVideoProjectStore.load(self.store.path)
        self.assertEqual(disk.project["tracking_runs"][0]["status"], "committed")
        self.service.rollback(run["id"])
        self.assertFalse(self.label(0).exists())

    def test_missing_rollback_backup_does_not_touch_any_accepted_label(self):
        for index in range(2):
            Annotation(str(self.frames[index]), str(self.label(index))).save_annotation()
        run = self.stage()
        self.commit(run)
        before = [self.label(index).read_bytes() for index in range(2)]
        manifest = self.store.tracking_directory / run["id"] / "accepted_backup/backup_manifest.json"
        entries = json.loads(manifest.read_text())
        backup = Path(entries[-1]["backup_path"])
        if not backup.is_absolute():
            backup = self.store.case_directory / backup
        backup.unlink()
        with self.assertRaises(ValueError):
            self.service.rollback(run["id"])
        self.assertEqual([self.label(index).read_bytes() for index in range(2)], before)

    def test_second_file_rollback_failure_compensates_first_file(self):
        for index in range(2):
            Annotation(str(self.frames[index]), str(self.label(index))).save_annotation()
        run = self.stage()
        self.commit(run)
        before = [self.label(index).read_bytes() for index in range(2)]
        from IVAS.video import tracking
        write = tracking.atomic_write_bytes
        def fail_second(path, data):
            if Path(path) == self.label(1):
                raise OSError("injected second-label failure")
            return write(path, data)
        with mock.patch.object(tracking, "atomic_write_bytes", side_effect=fail_second):
            with self.assertRaises(OSError):
                self.service.rollback(run["id"])
        self.assertEqual([self.label(index).read_bytes() for index in range(2)], before)
        self.assertEqual(self.service._run(run["id"])["status"], "committed")

    def test_reviewed_tracking_and_manual_edits_are_protected(self):
        run = self.stage()
        self.commit(run)
        annotation = Annotation(str(self.frames[0]), str(self.label(0))).load_annotation()
        annotation.objects[0].review_status = "reviewed"
        annotation.objects[0].segmentation[0] = [12, 12]
        annotation.save_annotation()
        run = self.stage()
        self.commit(run)
        restored = Annotation(str(self.frames[0]), str(self.label(0))).load_annotation()
        self.assertEqual(restored.objects[0].segmentation[0], [12, 12])
        run = self.stage()
        self.service.commit(run["id"], self.frames.__getitem__, self.label, replace_manual=True)
        replaced = Annotation(str(self.frames[0]), str(self.label(0))).load_annotation()
        self.assertNotEqual(replaced.objects[0].segmentation[0], [12, 12])

    def test_crash_recovery_uses_manifest_as_commit_decision(self):
        # Actual child process exit skips all context-manager compensation.
        script = '''
import os, sys
from pathlib import Path
from IVAS.video.project import UnifiedVideoProjectStore
from IVAS.video.transactions import label_transaction
store = UnifiedVideoProjectStore.load(Path(sys.argv[1]))
label = Path(sys.argv[2])
run_id, committed = sys.argv[3:5]
with label_transaction(store, run_id, "committed", [label]):
    label.write_bytes(b"new")
    if committed == "yes":
        store.project["tracking_runs"][0]["status"] = "committed"
        store.is_dirty = True
        store.save()
    os._exit(23)
'''
        run = self.stage()
        for committed in ("no", "yes"):
            with self.subTest(committed=committed):
                self.label(0).write_bytes(b"old")
                result = subprocess.run([sys.executable, "-c", script, str(self.store.path),
                                         str(self.label(0)), run["id"], committed],
                                        capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 23, result.stderr.decode(errors="replace"))
                UnifiedVideoProjectStore.load(self.store.path)
                self.assertEqual(self.label(0).read_bytes(), b"new" if committed == "yes" else b"old")
                self.assertEqual(list(self.store.tracking_directory.glob(".transaction-*")), [])


if __name__ == "__main__":
    unittest.main()
