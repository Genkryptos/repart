"""Score a rendered silhouette against an observed (SAM) mask, in image space."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Comparison:
    iou: float
    chamfer_px: float  # mean symmetric edge distance
    hausdorff_px: float  # 95th-percentile symmetric edge distance (robust)
    chamfer_mm: float
    hausdorff_mm: float
    area_ratio: float  # rendered / observed; > 1 means the model is too big

    def as_dict(self) -> dict:
        return asdict(self)


def soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def _edge_points(m: np.ndarray) -> np.ndarray:
    """Sub-pixel outline points: the 0.5 iso-contour of a soft mask (all contours, holes included)."""
    u8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    big = cv2.resize(u8, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    contours, _ = cv2.findContours((big >= 128).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2))
    return (np.vstack([c.reshape(-1, 2) for c in contours]) + 0.5) / 4.0


def compare(rendered: np.ndarray, observed: np.ndarray, mm_per_px: float) -> Comparison:
    """Both masks are full-frame, float 0..1, same shape."""
    pr, po = _edge_points(rendered), _edge_points(observed)
    if len(pr) == 0 or len(po) == 0:
        return Comparison(soft_iou(rendered, observed), np.inf, np.inf, np.inf, np.inf, 0.0)
    d1, _ = cKDTree(po).query(pr)
    d2, _ = cKDTree(pr).query(po)
    d = np.concatenate([d1, d2])
    chamfer, haus = float(d.mean()), float(np.percentile(d, 95))
    return Comparison(
        iou=soft_iou(rendered, observed),
        chamfer_px=chamfer,
        hausdorff_px=haus,
        chamfer_mm=chamfer * mm_per_px,
        hausdorff_mm=haus * mm_per_px,
        area_ratio=float(rendered.sum() / max(observed.sum(), 1e-9)),
    )


def residual_overlay(image_bgr: np.ndarray, rendered: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Green = matched, red = observed but not modelled (missing), blue = modelled but not observed (extra).

    This is the picture the agent looks at to decide topology edits.
    """
    out = image_bgr.astype(np.float32) * 0.5
    r, o = rendered > 0.5, observed > 0.5
    out[r & o] = out[r & o] * 0.4 + np.array([60, 200, 60]) * 0.6
    out[o & ~r] = np.array([40, 40, 230])
    out[r & ~o] = np.array([230, 120, 40])
    return out.clip(0, 255).astype(np.uint8)
