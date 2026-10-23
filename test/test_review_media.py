import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from PIL import Image

from IVAS.video.media import scan_frame_timestamps


class MediaReviewTests(unittest.TestCase):
    def test_nonzero_timestamp_origin_and_invalid_later_timestamps(self):
        class Capture:
            timestamps = [1000, 1040, 1080]
            def __init__(self, _): self.index = -1
            def isOpened(self): return True
            def get(self, key):
                return {cv2.CAP_PROP_FRAME_COUNT: 3, cv2.CAP_PROP_FPS: 25,
                        cv2.CAP_PROP_POS_MSEC: self.timestamps[max(0, self.index)]}.get(key, 0)
            def read(self):
                self.index += 1
                return (True, np.zeros((1, 1, 3), np.uint8)) if self.index < 3 else (False, None)
            def release(self): pass
        for values in ([1000, 1040, 1080], [1000, 1040, float("nan")], [1000, 1040, 0]):
            with self.subTest(values=values):
                Capture.timestamps = values
                mapping = scan_frame_timestamps(Path("fixture.mp4"), capture_factory=Capture)
                self.assertEqual(list(mapping.timestamps_ms), [0, 40, 80])
                self.assertEqual(mapping.frame_for_timestamp(40), 1)

    def test_video_loader_contract_for_cpu_and_each_sam2_compatible_backend(self):
        import torch
        from IVAS.segment_any.segment_any import SegAnyVideo
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (8, 6)).save(Path(directory) / "frame.png")
            for backend in ("sam2", "sam2.1", "sam3"):
                for asynchronous in (False, True):
                    with self.subTest(backend=backend, asynchronous=asynchronous):
                        facade = SegAnyVideo.__new__(SegAnyVideo)
                        facade.model_source, facade.device, facade.model_dtype = backend, "cpu", torch.float32
                        facade.predictor = SimpleNamespace(image_size=8, _get_image_feature=mock.Mock())
                        facade.reset_state = mock.Mock()
                        facade.init_state(directory, ["frame.png"], offload_video_to_cpu=False,
                                          offload_state_to_cpu=False, async_loading_frames=asynchronous)
                        state = facade.inference_state
                        self.assertEqual(state["num_frames"], 1)
                        self.assertEqual(str(state["storage_device"]), "cpu")
                        self.assertEqual(tuple(state["images"][0].shape), (3, 8, 8))
                        thread = getattr(state["images"], "thread", None)
                        if thread is not None:
                            thread.join(timeout=5)
