"""Fit a parametric SCAD model to an observed silhouette: named params + placement.

Everything is evaluated inside a crop around the observed part, so one evaluation
costs a cached OpenSCAD compile plus one small raster: ~0.05-0.1 s on 12 MP photos.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize

from .camera import Camera
from .compare import Comparison, compare, soft_iou
from .render import ScadError, scad_to_mesh, silhouette


@dataclass
class Observation:
    image: np.ndarray  # full BGR photo
    mask: np.ndarray  # full-frame soft part mask, 0..1
    cam: Camera
    crop: tuple[int, int, int, int] = (0, 0, 0, 0)  # x0, y0, x1, y1

    def __post_init__(self):
        if self.crop == (0, 0, 0, 0):
            x, y, w, h = cv2.boundingRect((self.mask > 0.5).astype(np.uint8))
            pad = max(40, int(0.15 * max(w, h)))
            H, W = self.mask.shape
            self.crop = (max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H))

    @property
    def mask_crop(self) -> np.ndarray:
        x0, y0, x1, y1 = self.crop
        return self.mask[y0:y1, x0:x1]

    @property
    def image_crop(self) -> np.ndarray:
        x0, y0, x1, y1 = self.crop
        return self.image[y0:y1, x0:x1]

    def tilt_deg(self) -> float:
        axis = self.cam.R.T @ np.array([0.0, 0.0, 1.0])
        return float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))

    def initial_xy(self, z_mm: float) -> np.ndarray:
        """Observed mask centroid back-projected onto the plane z = z_mm."""
        ys, xs = np.nonzero(self.mask > 0.5)
        ray = self.cam.R.T @ (np.linalg.inv(self.cam.K) @ np.array([xs.mean(), ys.mean(), 1.0]))
        C = self.cam.center
        return (C + (z_mm - C[2]) / ray[2] * ray)[:2]


def centre_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Spark is told to centre parts on the origin; enforce it so placement means the same thing."""
    m = mesh.copy()
    lo, hi = m.bounds
    m.vertices -= [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
    return m


def place(mesh: trimesh.Trimesh, x: float, y: float, yaw: float) -> trimesh.Trimesh:
    c, s = np.cos(yaw), np.sin(yaw)
    m = mesh.copy()
    m.vertices = m.vertices @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [x, y, 0]
    return m


def render_crop(mesh: trimesh.Trimesh, obs: Observation, supersample: int = 4) -> np.ndarray:
    soft, (bx0, by0, bx1, by1) = silhouette(mesh, obs.cam, supersample=supersample)
    x0, y0, x1, y1 = obs.crop
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    ix0, iy0, ix1, iy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
    if ix1 > ix0 and iy1 > iy0:
        out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = soft[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
    return out


@dataclass
class FitResult:
    params: dict[str, float]  # all params, tuned values substituted
    placement: tuple[float, float, float]  # x mm, y mm, yaw rad
    iou: float
    comparison: Comparison
    rendered: np.ndarray  # crop
    evals: int
    seconds: float
    changes: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (before, after)


def fit(
    code: str,
    params: dict[str, float],
    tunable: list[str],
    obs: Observation,
    placement: tuple[float, float, float] | None = None,
    max_evals: int = 500,
    prior_weight: float = 1e-4,
) -> FitResult:
    """Nelder-Mead on (tunable params, x, y, yaw) maximising silhouette IoU.

    A tiny quadratic pull toward the drafted values breaks ties along directions a
    single view can't see (e.g. taper vs. height), so unobservable params stay put.
    """
    t0 = time.perf_counter()
    tunable = [k for k in tunable if k in params]
    p0 = np.array([params[k] for k in tunable], dtype=float)
    scale = np.maximum(0.1 * np.abs(p0), 0.5)
    target = obs.mask_crop
    meshes: dict[tuple, trimesh.Trimesh | None] = {}
    n = {"evals": 0}

    def mesh_for(vals: np.ndarray) -> trimesh.Trimesh | None:
        key = tuple(np.round(vals, 3))
        if key not in meshes:
            p = dict(params) | {k: float(v) for k, v in zip(tunable, key)}
            try:
                meshes[key] = centre_mesh(scad_to_mesh(code, p)[0])
            except ScadError:
                meshes[key] = None
        return meshes[key]

    def loss(z: np.ndarray) -> float:
        n["evals"] += 1
        vals, (x, y, yaw) = z[: len(tunable)], z[len(tunable):]
        if np.any(vals <= 0):
            return 2.0
        m = mesh_for(vals)
        if m is None:
            return 2.0
        iou = soft_iou(render_crop(place(m, x, y, yaw), obs, supersample=3), target)
        return 1.0 - iou + prior_weight * float(np.sum(((vals - p0) / scale) ** 2))

    if placement is None:
        m0 = mesh_for(p0)
        if m0 is None:
            raise ScadError("Draft does not compile with its own parameters.")
        xy = obs.initial_xy(m0.bounds[1][2] / 2)
        yaws = np.radians(np.arange(0, 360, 15))
        yaw = min(yaws, key=lambda a: loss(np.r_[p0, xy, a]))
        placement = (float(xy[0]), float(xy[1]), float(yaw))

    z0 = np.r_[p0, placement]
    steps = np.r_[scale * 0.5, 0.75, 0.75, np.radians(4)]
    res = minimize(loss, z0, method="Nelder-Mead",
                   options={"initial_simplex": np.vstack([z0, z0 + np.diag(steps)]),
                            "xatol": 2e-3, "fatol": 1e-6, "maxfev": max_evals, "adaptive": len(z0) > 4})
    vals, (x, y, yaw) = res.x[: len(tunable)], res.x[len(tunable):]
    final = dict(params) | {k: round(float(v), 3) for k, v in zip(tunable, vals)}
    m = mesh_for(np.round(vals, 3))
    rendered = render_crop(place(m, x, y, yaw), obs)
    mm_px = obs.cam.mm_per_px_at([x, y, m.bounds[1][2] / 2])
    return FitResult(
        params=final,
        placement=(float(x), float(y), float(yaw)),
        iou=soft_iou(rendered, target),
        comparison=compare(rendered, target, mm_px),
        rendered=rendered,
        evals=n["evals"],
        seconds=time.perf_counter() - t0,
        changes={k: (params[k], final[k]) for k in tunable if abs(final[k] - params[k]) >= 0.005},
    )


def sector_residuals(rendered: np.ndarray, observed: np.ndarray, mm_per_px: float, sectors: int = 12) -> list[dict]:
    """Where the model is missing material (observed, not rendered) or has extra, by clock
    position in the photo crop, as an average thickness in mm. Small = alignment noise."""
    o, r = observed > 0.5, rendered > 0.5
    ys, xs = np.nonzero(o)
    cy, cx = ys.mean(), xs.mean()
    perim = cv2.arcLength(max(cv2.findContours(o.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea), True)
    arc = perim / sectors
    out = []
    for kind, m in (("missing", o & ~r), ("extra", r & ~o)):
        py, px = np.nonzero(m)
        if len(px) == 0:
            continue
        # Clock position: 12 o'clock is image-up, clockwise.
        ang = (np.degrees(np.arctan2(px - cx, -(py - cy))) + 360) % 360
        idx = (ang // (360 / sectors)).astype(int)
        counts = np.bincount(idx, minlength=sectors)
        for s, c in enumerate(counts):
            t = c / arc * mm_per_px
            if t >= 0.05:
                out.append({"kind": kind, "clock": 12 if s == 0 else s, "thickness_mm": round(float(t), 2)})
    return sorted(out, key=lambda d: -d["thickness_mm"])
