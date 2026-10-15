import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from IVAS.segment_any.precision import apply_precision, normalize_precision
from IVAS.segment_any.sam3.build_sam import Sam3Predictor, Sam31Predictor
from IVAS.segment_any.sam3.model_builder import (
    _extract_sam31_detector_state_dict,
)


class PrecisionPolicyTests(unittest.TestCase):
    def test_legacy_boolean_is_only_a_fallback(self):
        self.assertEqual(normalize_precision(use_bfloat16=True), "bfloat16")
        self.assertEqual(normalize_precision(use_bfloat16=False), "float32")
        self.assertEqual(
            normalize_precision("float16", use_bfloat16=True), "float16"
        )

    def test_explicit_cast_preserves_complex_and_integer_buffers(self):
        module = torch.nn.Linear(2, 2)
        module.register_buffer(
            "rope", torch.tensor([1 + 2j], dtype=torch.complex64)
        )
        module.register_buffer("indices", torch.tensor([1], dtype=torch.int64))

        effective = apply_precision(module, "bfloat16")

        self.assertEqual(effective, torch.bfloat16)
        self.assertEqual(module.weight.dtype, torch.bfloat16)
        self.assertEqual(module.rope.dtype, torch.complex64)
        self.assertEqual(module.indices.dtype, torch.int64)

    def test_original_preserves_mixed_checkpoint_tensors(self):
        module = torch.nn.Linear(2, 2)
        module.bias.data = module.bias.data.to(torch.float16)
        module._checkpoint_dtype = torch.float32

        effective = apply_precision(module, "original")

        self.assertEqual(effective, torch.float32)
        self.assertEqual(module.weight.dtype, torch.float32)
        self.assertEqual(module.bias.dtype, torch.float16)


class Sam31CheckpointMappingTests(unittest.TestCase):
    def setUp(self):
        self.model_state = {
            "backbone.weight": torch.zeros(2, 2),
            "head.bias": torch.zeros(2),
        }

    def test_bare_detector_layout(self):
        checkpoint = {
            "backbone.weight": torch.ones(2, 2),
            "head.bias": torch.ones(2),
        }
        mapped = _extract_sam31_detector_state_dict(
            checkpoint, self.model_state
        )
        self.assertEqual(set(mapped), set(self.model_state))

    def test_official_layout_ignores_only_tracker_branch(self):
        checkpoint = {
            "sam3_model.backbone.weight": torch.ones(2, 2),
            "sam3_model.head.bias": torch.ones(2),
            "sam2_predictor.tracker.weight": torch.ones(1),
        }
        mapped = _extract_sam31_detector_state_dict(
            checkpoint, self.model_state
        )
        self.assertEqual(set(mapped), set(self.model_state))

    def test_missing_learned_key_fails_fast(self):
        with self.assertRaises(RuntimeError):
            _extract_sam31_detector_state_dict(
                {"backbone.weight": torch.ones(2, 2)}, self.model_state
            )

    def test_shape_mismatch_fails_fast(self):
        checkpoint = {
            "backbone.weight": torch.ones(3, 2),
            "head.bias": torch.ones(2),
        }
        with self.assertRaises(RuntimeError):
            _extract_sam31_detector_state_dict(
                checkpoint, self.model_state
            )


class PredictorRegressionTests(unittest.TestCase):
    class _Processor:
        def set_image(self, image, state=None):
            if image is None:
                raise AssertionError("cached visual prediction re-encoded the image")
            return {
                "masks": torch.ones(1, 1, 2, 2, dtype=torch.bool),
                "scores": torch.tensor([0.9]),
            }

        def reset_all_prompts(self, state):
            return None

        def add_geometric_prompt(self, state, box, label):
            state["masks"] = torch.ones(1, 1, 2, 2, dtype=torch.bool)
            state["scores"] = torch.tensor([0.9])
            return state

    def test_sam3_visual_predictor_keeps_legacy_contract(self):
        predictor = object.__new__(Sam3Predictor)
        predictor._sam3_processor = self._Processor()

        masks, scores = predictor.predict_with_visual_prompt(
            Image.new("RGB", (2, 2)), [[0, 0, 2, 2]], [True]
        )

        self.assertEqual(masks.shape, (1, 2, 2))
        self.assertEqual(tuple(scores.shape), (1,))

    def test_sam3_reset_marks_the_image_unset(self):
        interactive = SimpleNamespace(
            _is_image_set=True,
            _orig_hw=[(2, 2)],
            _features={"image_embed": torch.ones(1)},
        )
        predictor = object.__new__(Sam3Predictor)
        predictor.model = SimpleNamespace(inst_interactive_predictor=interactive)

        predictor.reset_image()

        self.assertFalse(interactive._is_image_set)
        self.assertIsNone(interactive._orig_hw)
        self.assertIsNone(interactive._features)

    def test_sam31_visual_predictor_reuses_cached_image(self):
        predictor = object.__new__(Sam31Predictor)
        predictor.device = "cpu"
        predictor.model_dtype = torch.float32
        predictor._lock = threading.RLock()
        predictor._sam3_processor = self._Processor()
        predictor._cached_state = {
            "original_height": 2,
            "original_width": 2,
            "backbone_out": {},
        }

        masks, scores = predictor.predict_with_visual_prompt(
            None, [[0, 0, 2, 2]], [True]
        )

        np.testing.assert_array_equal(masks, np.ones((1, 2, 2), dtype=bool))
        self.assertEqual(scores.shape, (1,))


if __name__ == "__main__":
    unittest.main()
