"""Interactive SAM jobs: immutable prompts in, masks out; no Qt scene access."""

import numpy as np
from PyQt5 import QtCore


def restore_cached_image(segany, cached):
    """Caller holds the model's inference lock through restoration/prediction."""
    features = cached["features"]
    original_size = cached["original_size"]
    if segany.model_source == "sam3.1":
        segany.predictor.set_cached_state(features)
        return
    if segany.model_source == "sam_hq":
        features, segany.predictor.interm_features = features
    predictor = segany.predictor
    predictor.features = features
    predictor.original_size = original_size
    predictor.input_size = cached["input_size"]
    predictor.is_image_set = True
    predictor._orig_hw = list(original_size)
    predictor._features = features
    predictor._is_image_set = True
    if segany.model_source == "sam3":
        interactive = predictor.model.inst_interactive_predictor
        interactive._is_image_set = True
        interactive._orig_hw = [original_size]
        interactive._features = features


# Retain parentless workers if a scene is deleted while cancellation completes.
_active_workers = set()


class MaskPredictionWorker(QtCore.QThread):
    ready = QtCore.pyqtSignal(int, object, str)

    def __init__(self, request_id, segany, cached, points, labels, box):
        super().__init__()
        self.request_id = request_id
        self.segany = segany
        self.cached = dict(cached)
        self.points = points
        self.labels = labels
        self.box = box
        _active_workers.add(self)
        self.finished.connect(self._release)

    def _release(self):
        _active_workers.discard(self)
        self.deleteLater()

    def run(self):
        try:
            with self.segany.inference_lock:
                if self.isInterruptionRequested():
                    return
                restore_cached_image(self.segany, self.cached)
                if self.points:
                    mask = self.segany.predict_with_point_prompt(
                        [list(point) for point in self.points], list(self.labels)
                    )
                else:
                    mask = self.segany.predict_with_box_prompt(np.asarray(self.box))
            if not self.isInterruptionRequested():
                self.ready.emit(self.request_id, mask, "")
        except Exception as error:
            if not self.isInterruptionRequested():
                self.ready.emit(self.request_id, None, str(error))
