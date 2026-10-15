import copy
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5 import QtWidgets

from IVAS.widgets.video_annotation_dialogs import (
    TrackingReviewDialog,
    compose_tracking_overlay,
    tracking_review_colour,
)


class TrackingReviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_overlay_is_deterministic_and_does_not_mutate_inputs(self):
        frame = np.full((5, 6, 3), 80, dtype=np.uint8)
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[1:4, 2:5] = 255
        original_frame = frame.copy()
        original_mask = mask.copy()

        rendered = compose_tracking_overlay(frame, [("object-7", mask)], alpha=0.5)

        self.assertTrue(np.array_equal(frame, original_frame))
        self.assertTrue(np.array_equal(mask, original_mask))
        self.assertTrue(np.array_equal(rendered[0, 0], frame[0, 0]))
        colour = np.asarray(tracking_review_colour("object-7"), dtype=np.float32)
        expected = np.rint(
            frame[2, 3].astype(np.float32) * 0.5 + colour * 0.5
        ).astype(np.uint8)
        self.assertTrue(np.array_equal(rendered[2, 3], expected))
        self.assertTrue(
            np.array_equal(
                rendered,
                compose_tracking_overlay(frame, [("object-7", mask)], alpha=0.5),
            )
        )

    def test_opening_and_cancelling_review_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            frame[:, :, 1] = 90
            mask = np.zeros((24, 32), dtype=np.uint8)
            mask[5:18, 8:24] = 255
            frame_path = root / "frame.png"
            mask_path = root / "mask.png"
            self.assertTrue(cv2.imwrite(str(frame_path), frame))
            self.assertTrue(cv2.imwrite(str(mask_path), mask))
            label_path = root / "label.json"
            label_path.write_text('{"sentinel": true}', encoding="utf-8")
            run = {
                "id": "track-run",
                "direction": "forward",
                "start_frame": 3,
                "end_frame": 3,
                "status": "staged",
                "staged_frames": [
                    {
                        "frame_index": 3,
                        "objects": [
                            {
                                "object_id": "object-7",
                                "mask_path": mask_path.name,
                                "review_status": "draft",
                            }
                        ],
                    }
                ],
            }
            before_run = copy.deepcopy(run)
            before_label = label_path.read_bytes()

            dialog = TrackingReviewDialog(
                run,
                lambda _index: cv2.imread(str(frame_path), cv2.IMREAD_COLOR),
                lambda value: root / value,
                {"object-7": {"category": "Lens", "track_id": "track-7"}},
            )
            self.app.processEvents()

            self.assertFalse(dialog._original_pixmap.isNull())
            self.assertFalse(dialog._overlay_pixmap.isNull())
            self.assertIn("Lens", dialog.legend_label.text())
            self.assertEqual(dialog.error_label.text(), "")
            dialog.reject()
            self.assertEqual(dialog.result(), QtWidgets.QDialog.Rejected)
            self.assertEqual(run, before_run)
            self.assertEqual(label_path.read_bytes(), before_label)
            dialog.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
