"""Object tracks: per-frame centroid, area and box in full-frame pixel coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int


def video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    info = VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


@dataclass
class Track:
    object_id: str
    concept: str
    frames: np.ndarray  # int frame indices (-1 for a still image)
    cx: np.ndarray  # px
    cy: np.ndarray  # px
    area: np.ndarray  # px^2
    bbox: np.ndarray  # (n, 4) left, top, right, bottom

    def __len__(self) -> int:
        return len(self.frames)

    def times(self, fps: float) -> np.ndarray:
        return self.frames / fps

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "concept": self.concept,
            "frames": self.frames.tolist(),
            "cx": self.cx.tolist(),
            "cy": self.cy.tolist(),
            "area": self.area.tolist(),
            "bbox": self.bbox.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Track:
        return cls(
            object_id=d["object_id"],
            concept=d["concept"],
            frames=np.asarray(d["frames"], dtype=int),
            cx=np.asarray(d["cx"], dtype=float),
            cy=np.asarray(d["cy"], dtype=float),
            area=np.asarray(d["area"], dtype=float),
            bbox=np.asarray(d["bbox"], dtype=float).reshape(-1, 4),
        )


def save_tracks(path: Path, tracks: list[Track]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([t.to_dict() for t in tracks]))


def load_tracks(path: Path) -> list[Track]:
    return [Track.from_dict(d) for d in json.loads(path.read_text())]


def mask_stats(
    raster: np.ndarray, left: float, top: float, right: float, bottom: float
) -> tuple[float, float, float] | None:
    """Centroid and area of a box-local mask, mapped to full-frame pixels.

    SAM returns each mask cropped to its bounds and possibly resampled, so mask
    pixel (row, col) covers a (bw/w) x (bh/h) patch of the frame.
    """
    h, w = raster.shape
    ys, xs = np.nonzero(raster)
    if len(xs) == 0:
        return None
    sx = (right - left) / w
    sy = (bottom - top) / h
    cx = left + (xs.mean() + 0.5) * sx
    cy = top + (ys.mean() + 0.5) * sy
    return cx, cy, len(xs) * sx * sy


def track_color(
    video: str | Path,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    concept: str,
    min_area: float = 20,
) -> Track:
    """Classical-CV baseline: largest blob inside an HSV range, per frame."""
    cap = cv2.VideoCapture(str(video))
    frames, cx, cy, area, bbox = [], [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            a = stats[k, cv2.CC_STAT_AREA]
            if a >= min_area:
                x, y, bw, bh = stats[k, :4]
                frames.append(i)
                cx.append(cents[k, 0] + 0.5)
                cy.append(cents[k, 1] + 0.5)
                area.append(float(a))
                bbox.append([x, y, x + bw, y + bh])
        i += 1
    cap.release()
    return Track(
        object_id="0",
        concept=concept,
        frames=np.asarray(frames, dtype=int),
        cx=np.asarray(cx),
        cy=np.asarray(cy),
        area=np.asarray(area),
        bbox=np.asarray(bbox, dtype=float).reshape(-1, 4),
    )
