"""Model-backed smoke for staged SAM video tracking and transactional labels."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from threading import Event

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IVAS.annotation import Annotation
from IVAS.segment_any.segment_any import SegAnyVideo
from IVAS.video.project import UnifiedVideoProjectStore
from IVAS.video.tracking import (
    SegAnyTrackingAdapter,
    TrackingRequest,
    TrackingSeed,
    VideoTrackingService,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def write_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_frames(root: Path, count: int = 5):
    root.mkdir(parents=True, exist_ok=True)
    frames = {}
    for index in range(count):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        x = 28 + index * 3
        image[36:88, x : x + 42, 1] = 210
        path = root / "frame_{:012d}.png".format(index)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("Unable to create synthetic tracking frame")
        frames[index] = path
    seed = np.zeros((128, 128), dtype=np.uint8)
    seed[36:88, 28:70] = 255
    seed_path = root.parent / "seed.png"
    if not cv2.imwrite(str(seed_path), seed):
        raise RuntimeError("Unable to create tracking seed")
    return frames, seed_path


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    metrics = {
        "status": "failed",
        "engine": "sam-video",
        "checkpoint": checkpoint.name,
        "precision": args.precision,
        "checks": {
            "staged_not_direct_write": False,
            "manual_wins": False,
            "rollback_restores": False,
        },
        "outputs": {"nonempty_tracked_frames": 0, "staged_frames": 0},
    }
    facade = None
    work_root = args.output.parent / "tracking-smoke-workspace"
    try:
        import torch

        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this smoke profile")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if work_root.exists():
            shutil.rmtree(work_root)
        frames, seed_path = create_frames(work_root / "frames")
        media = work_root / "synthetic.mp4"
        media.write_bytes(b"synthetic tracking smoke media identity")
        store = UnifiedVideoProjectStore.create(
            work_root / "case" / "project.ivas.json",
            media,
            duration_ms=200,
            fps=25,
            frame_count=len(frames),
            width=128,
            height=128,
        )
        instance = store.ensure_instance(
            "instrument",
            "Synthetic instrument",
            object_id="instrument-1",
            track_id="track-instrument-1",
            review_status="reviewed",
        )
        store.save()
        facade = SegAnyVideo(str(checkpoint), precision=args.precision)
        service = VideoTrackingService(store)
        run = service.stage(
            TrackingRequest(
                direction="bidirectional",
                start_frame=0,
                end_frame=len(frames) - 1,
                seeds=[TrackingSeed(0, instance["object_id"], seed_path)],
                engine="segany-video",
            ),
            SegAnyTrackingAdapter(facade),
            frames,
            Event(),
        )
        labels_before = list(store.labels_directory.glob("*.json"))
        metrics["checks"]["staged_not_direct_write"] = not labels_before
        staged_frames = run.get("staged_frames", [])
        metrics["outputs"]["staged_frames"] = len(staged_frames)
        metrics["outputs"]["nonempty_tracked_frames"] = sum(
            bool(frame.get("objects")) for frame in staged_frames
        )

        def image_path(index):
            return frames[index]

        def label_path(index):
            return store.labels_directory / "frame_{:012d}.json".format(index)

        service.commit(run["id"], image_path, label_path)
        committed = Annotation(str(frames[0]), str(label_path(0))).load_annotation()
        metrics["checks"]["manual_wins"] = bool(committed.objects)
        service.rollback(run["id"])
        metrics["checks"]["rollback_restores"] = not any(
            store.labels_directory.glob("*.json")
        )
        all_required = (
            all(metrics["checks"].values())
            and metrics["outputs"]["nonempty_tracked_frames"] > 0
        )
        metrics["checks"]["all_required"] = all_required
        metrics["status"] = "passed" if all_required else "failed"
        if torch.cuda.is_available():
            metrics["gpu"] = {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        write_metrics(args.output, metrics)
        return 0 if all_required else 1
    except Exception as error:
        metrics["error"] = "{}: {}".format(type(error).__name__, error)
        write_metrics(args.output, metrics)
        raise
    finally:
        if facade is not None:
            try:
                facade.release()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
