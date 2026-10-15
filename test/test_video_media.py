import tempfile
import unittest
from pathlib import Path
from threading import Event

import numpy as np

from IVAS.video.media import FrameTimestampMap, OperationCancelled, scan_frame_timestamps


class _FakeCapture:
    def __init__(self, timestamps):
        self.timestamps = list(timestamps)
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.timestamps):
            return False, None
        self.index += 1
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.timestamps)
        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps[max(0, self.index - 1)]
        return 0

    def release(self):
        self.released = True


class VideoMediaTests(unittest.TestCase):
    def test_vfr_map_nearest_lookup_and_round_trip(self):
        capture = _FakeCapture([0, 33, 100, 142, 230])
        mapping = scan_frame_timestamps(
            Path("case.mp4"), capture_factory=lambda _path: capture
        )
        self.assertEqual(mapping.method, "decoded_pts")
        self.assertEqual(mapping.frame_for_timestamp(90), 2)
        self.assertEqual(mapping.timestamp_for_frame(3), 142)
        self.assertTrue(capture.released)
        with tempfile.TemporaryDirectory() as directory:
            path = mapping.save(Path(directory) / "pts.json")
            self.assertEqual(FrameTimestampMap.load(path), mapping)

    def test_duplicate_codec_timestamps_are_marked_mixed(self):
        mapping = scan_frame_timestamps(
            Path("case.mp4"), capture_factory=lambda _path: _FakeCapture([0, 0, 80])
        )
        self.assertEqual(mapping.method, "mixed")
        self.assertEqual(mapping.timestamps_ms, [0, 40, 80])

    def test_scan_honours_cancellation(self):
        cancelled = Event()
        cancelled.set()
        capture = _FakeCapture([0, 40])
        with self.assertRaises(OperationCancelled):
            scan_frame_timestamps(
                Path("case.mp4"),
                cancel_event=cancelled,
                capture_factory=lambda _path: capture,
            )
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()

