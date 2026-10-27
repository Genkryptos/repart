"""Fit a parametric SCAD model to an observed silhouette: named params + placement.

Everything is evaluated inside a crop around the observed part, so one evaluation
costs a cached OpenSCAD compile plus one small raster: ~0.05-0.1 s on 12 MP photos.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
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


Placement = tuple[float, float, float]  # x mm, y mm, yaw rad on the sheet

_POSITION = re.compile(r"(offset|shift|pos|centre|center|^c[xyz]|_c[xyz]$|^[xyz]\d*$|^[xyz]_|_[xyz]$|_[xyz]\d$)", re.I)


def is_position(name: str) -> bool:
    """Heuristic: does this parameter place something (may be negative) rather than size it?"""
    return bool(_POSITION.search(name))


@dataclass
class FitResult:
    params: dict[str, float]  # all params, tuned values substituted
    placements: list[Placement]  # one per view
    ious: list[float]
    comparisons: list[Comparison]
    rendered: list[np.ndarray]  # crops, one per view
    evals: int
    seconds: float
    changes: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (before, after)

    @property
    def iou(self) -> float:
        return float(np.mean(self.ious))

    @property
    def chamfer_mm(self) -> float:
        return float(np.mean([c.chamfer_mm for c in self.comparisons]))


def fit(
    code: str,
    params: dict[str, float],
    tunable: list[str],
    views: Observation | list[Observation],
    placements: list[Placement] | None = None,
    max_evals: int | None = None,
    prior_weight: float = 1e-4,
    method: str = "nelder-mead",  # "cma" is ~4x faster but collapsed the bracket wall in testing
    _retry: bool = True,
) -> FitResult:
    """Nelder-Mead on (tunable params, then x, y, yaw per view) maximising mean IoU.

    Shape params are shared; each photo gets its own placement, so the part may be
    moved between shots. A tiny quadratic pull toward the drafted values breaks ties
    along directions no view can see, so unobservable params stay put.
    """
    t0 = time.perf_counter()
    views = [views] if isinstance(views, Observation) else list(views)
    tunable = [k for k in tunable if k in params]
    k = len(tunable)
    p0 = np.array([params[t] for t in tunable], dtype=float)
    scale = np.maximum(0.1 * np.abs(p0), 0.5)
    # Sizes must stay positive; positions (offsets, centres) may cross zero, e.g. when
    # holes were drafted on the wrong side of a wall.
    sized = np.array([not is_position(t) for t in tunable], dtype=bool)
    meshes: dict[tuple, trimesh.Trimesh | None] = {}
    n = {"evals": 0}

    def mesh_for(vals: np.ndarray) -> trimesh.Trimesh | None:
        key = tuple(np.round(vals, 3))
        if key not in meshes:
            p = dict(params) | {t: float(v) for t, v in zip(tunable, key)}
            try:
                meshes[key] = centre_mesh(scad_to_mesh(code, p)[0])
            except ScadError:
                meshes[key] = None
        return meshes[key]

    def view_loss(m: trimesh.Trimesh, pl, obs: Observation) -> float:
        return 1.0 - soft_iou(render_crop(place(m, *pl), obs, supersample=3), obs.mask_crop)

    def loss(z: np.ndarray) -> float:
        n["evals"] += 1
        vals = z[:k]
        if np.any(vals[sized] <= 0):
            return 2.0
        m = mesh_for(vals)
        if m is None:
            return 2.0
        pls = z[k:].reshape(-1, 3)
        err = np.mean([view_loss(m, pl, obs) for pl, obs in zip(pls, views)])
        return float(err) + prior_weight * float(np.sum(((vals - p0) / scale) ** 2))

    if placements is None:
        m0 = mesh_for(p0)
        if m0 is None:
            raise ScadError("Draft does not compile with its own parameters.")
        placements = []
        for obs in views:
            # Per-view start: centroid back-projection, yaw grid, then a short 3-D polish.
            xy = obs.initial_xy(m0.bounds[1][2] / 2)
            yaw = min(np.radians(np.arange(0, 360, 15)), key=lambda a: view_loss(m0, (*xy, a), obs))
            z = np.r_[xy, yaw]
            res = minimize(lambda q: view_loss(m0, q, obs), z, method="Nelder-Mead",
                           options={"initial_simplex": np.vstack([z, z + np.diag([0.75, 0.75, np.radians(4)])]),
                                    "xatol": 1e-2, "fatol": 1e-5, "maxfev": 80})
            n["evals"] += res.nfev
            placements.append(tuple(float(v) for v in res.x))

    z0 = np.r_[p0, np.ravel(placements)]
    steps = np.r_[scale * 0.5, np.tile([0.75, 0.75, np.radians(4)], len(views))]
    if k and method == "cma":
        x_best = _cma(loss, lambda Z: _prefetch(mesh_for, [z[:k] for z in Z]), z0, steps,
                      max_evals or 150 + 60 * len(z0))
        res = minimize(loss, x_best, method="Nelder-Mead",  # short polish from the CMA optimum
                       options={"initial_simplex": np.vstack([x_best, x_best + np.diag(steps * 0.15)]),
                                "xatol": 1e-3, "fatol": 1e-6, "maxfev": 40 * len(z0), "adaptive": True})
    else:
        res = minimize(loss, z0, method="Nelder-Mead",
                       options={"initial_simplex": np.vstack([z0, z0 + np.diag(steps)]),
                                "xatol": 2e-3, "fatol": 1e-6, "maxfev": max_evals or 350 + 150 * len(z0),
                                "adaptive": len(z0) > 4})
    vals = res.x[:k]
    pls = [tuple(float(v) for v in pl) for pl in res.x[k:].reshape(-1, 3)]
    final = dict(params) | {t: round(float(v), 3) for t, v in zip(tunable, vals)}
    m = mesh_for(np.round(vals, 3))
    rendered, ious, comps = [], [], []
    for pl, obs in zip(pls, views):
        r = render_crop(place(m, *pl), obs)
        mm_px = obs.cam.mm_per_px_at([pl[0], pl[1], m.bounds[1][2] / 2])
        rendered.append(r)
        ious.append(soft_iou(r, obs.mask_crop))
        comps.append(compare(r, obs.mask_crop, mm_px))
    result = FitResult(
        params=final,
        placements=pls,
        ious=ious,
        comparisons=comps,
        rendered=rendered,
        evals=n["evals"],
        seconds=time.perf_counter() - t0,
        changes={t: (params[t], final[t]) for t in tunable if abs(final[t] - params[t]) >= 0.005},
    )
    # A structural dimension shrinking to almost nothing usually means the feature was
    # misplaced, not absent: from a bad start, deleting a wall is the quickest way to cut
    # "extra" area. Retry with it restored, the other fitted values kept, and placements
    # searched afresh, and keep whichever fit is better.
    collapsed = [t for t, v0, sz in zip(tunable, p0, sized) if sz and v0 > 1.0 and final[t] < 0.3 * v0]
    if collapsed and _retry:
        again = fit(code, final | {t: params[t] for t in collapsed}, tunable, views,
                    max_evals=max_evals, prior_weight=prior_weight, method=method, _retry=False)
        if again.iou > result.iou:
            again.changes = {t: (params[t], again.params[t]) for t in tunable
                             if abs(again.params[t] - params[t]) >= 0.005}
            again.evals += result.evals
            again.seconds = time.perf_counter() - t0
            return again
    return result


_POOL = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2))


def _prefetch(mesh_for, shape_vectors) -> None:
    """Compile a batch of candidate shapes in parallel: OpenSCAD runs as a subprocess, so
    threads overlap the ~50 ms process start-up that dominates each evaluation."""
    seen, todo = set(), []
    for v in shape_vectors:
        key = tuple(np.round(v, 3))
        if key not in seen:
            seen.add(key)
            todo.append(np.asarray(key))
    list(_POOL.map(mesh_for, todo))


def _cma(loss, prefetch, z0: np.ndarray, steps: np.ndarray, budget: int) -> np.ndarray:
    """CMA-ES in step-normalised coordinates. Proposes a population per generation, which
    is what lets compiles run in parallel, and explores more widely than Nelder-Mead."""
    import cma

    es = cma.CMAEvolutionStrategy(np.zeros_like(z0), 1.0,
                                  {"maxfevals": budget, "tolfun": 1e-6, "tolx": 1e-4,
                                   "verbose": -9, "seed": 7})
    while not es.stop():
        xs = es.ask()
        zs = [z0 + x * steps for x in xs]
        prefetch(zs)
        es.tell(xs, [loss(z) for z in zs])
    return z0 + np.asarray(es.result.xbest) * steps


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
