"""Presentation-timestamp-aware video indexing and cancellable decoding."""

from __future__ import annotations

import bisect
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, List, Optional, Tuple

import cv2


class OperationCancelled(RuntimeError):
    pass


ProgressCallback = Callable[[int, int], None]


@dataclass
class FrameTimestampMap:
    timestamps_ms: List[int]
    method: str = "decoded_pts"

    def __post_init__(self) -> None:
        values = [max(0, int(value)) for value in self.timestamps_ms]
        if any(right < left for left, right in zip(values, values[1:])):
            raise ValueError("Frame timestamps must be monotonic")
        self.timestamps_ms = values
        if self.method not in {"decoded_pts", "fps_fallback", "mixed"}:
            raise ValueError("Unknown frame timestamp method: {}".format(self.method))

    def __len__(self) -> int:
        return len(self.timestamps_ms)

    @property
    def duration_ms(self) -> int:
        return self.timestamps_ms[-1] if self.timestamps_ms else 0

    def timestamp_for_frame(self, frame_index: int) -> int:
        if not self.timestamps_ms:
            return 0
        index = min(len(self.timestamps_ms) - 1, max(0, int(frame_index)))
        return self.timestamps_ms[index]

    def frame_for_timestamp(self, timestamp_ms: int) -> int:
        if not self.timestamps_ms:
            return 0
        target = max(0, int(timestamp_ms))
        right = bisect.bisect_left(self.timestamps_ms, target)
        if right <= 0:
            return 0
        if right >= len(self.timestamps_ms):
            return len(self.timestamps_ms) - 1
        left = right - 1
        return left if target - self.timestamps_ms[left] <= self.timestamps_ms[right] - target else right

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "method": self.method,
            "timestamps_ms": self.timestamps_ms,
        }

    def save(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".{}-".format(destination.stem),
            suffix=".tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                json.dump(self.to_dict(), stream, ensure_ascii=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return destination

    @classmethod
    def load(cls, path: Path) -> "FrameTimestampMap":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported frame timestamp map")
        values = payload.get("timestamps_ms")
        if not isinstance(values, list):
            raise ValueError("Frame timestamp map is missing timestamps_ms")
        return cls(values, str(payload.get("method") or "decoded_pts"))

    @classmethod
    def from_fps(cls, frame_count: int, fps: float) -> "FrameTimestampMap":
        count = max(0, int(frame_count))
        rate = float(fps)
        if rate <= 0:
            raise ValueError("A positive FPS is required for fallback indexing")
        return cls(
            [int(round(index * 1000.0 / rate)) for index in range(count)],
            "fps_fallback",
        )


def scan_frame_timestamps(
    video_path: Path,
    cancel_event: Optional[Event] = None,
    progress: Optional[ProgressCallback] = None,
    capture_factory: Optional[Callable[[str], object]] = None,
) -> FrameTimestampMap:
    """Decode the stream once and collect presentation timestamps.

    OpenCV exposes backend presentation time through ``CAP_PROP_POS_MSEC``.
    When a codec/backend omits or repeats it, only those frames use a local FPS
    fallback and the map is labelled ``mixed`` instead of claiming exact PTS.
    """

    source = Path(video_path)
    capture = (capture_factory or cv2.VideoCapture)(str(source))
    try:
        if not capture.isOpened():
            raise ValueError("Unable to decode video: {}".format(source))
        total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fallback_step = 1000.0 / fps if fps > 0 else 40.0
        timestamps: List[int] = []
        fallback_used = False
        origin = None
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled("Video timestamp scan was cancelled")
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            index = len(timestamps)
            raw_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if not math.isfinite(raw_ms):
                raw_ms = 0.0 if origin is None else origin
                fallback_used = True
            if origin is None:
                origin = raw_ms
            candidate = max(0, int(round(raw_ms - origin)))
            if index and candidate <= timestamps[-1]:
                candidate = max(
                    timestamps[-1] + 1,
                    int(round(timestamps[-1] + fallback_step)),
                )
                fallback_used = True
            timestamps.append(candidate)
            if progress is not None and (index % 16 == 0 or index + 1 == total):
                progress(index + 1, total)
        if not timestamps:
            raise ValueError("Video contains no decodable frames: {}".format(source))
        method = "mixed" if fallback_used else "decoded_pts"
        return FrameTimestampMap(timestamps, method)
    finally:
        capture.release()


def decode_frame(
    video_path: Path,
    frame_index: int,
    cancel_event: Optional[Event] = None,
    capture_factory: Optional[Callable[[str], object]] = None,
):
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled("Frame extraction was cancelled")
    capture = (capture_factory or cv2.VideoCapture)(str(Path(video_path)))
    try:
        if not capture.isOpened():
            raise ValueError("Unable to open video for frame extraction")
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = capture.read()
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled("Frame extraction was cancelled")
        if not ok or frame is None:
            raise ValueError("Unable to decode frame {}".format(frame_index))
        return frame
    finally:
        capture.release()


def atomic_write_png(path: Path, frame) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(".png", frame)
    if not encoded_ok:
        raise ValueError("Unable to encode decoded frame")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{}-".format(destination.stem),
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


def decode_frame_range(
    video_path: Path,
    frame_indices: Iterable[int],
    output_directory: Path,
    cancel_event: Optional[Event] = None,
    progress: Optional[ProgressCallback] = None,
    capture_factory: Optional[Callable[[str], object]] = None,
) -> List[Path]:
    indices = [max(0, int(value)) for value in frame_indices]
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    capture = (capture_factory or cv2.VideoCapture)(str(Path(video_path)))
    written: List[Path] = []
    try:
        if not capture.isOpened():
            raise ValueError("Unable to open video for range decoding")
        for position, frame_index in enumerate(indices, start=1):
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled("Frame range decoding was cancelled")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError("Unable to decode frame {}".format(frame_index))
            path = destination / "frame_{:012d}.png".format(frame_index)
            atomic_write_png(path, frame)
            written.append(path)
            if progress is not None:
                progress(position, len(indices))
        return written
    finally:
        capture.release()
