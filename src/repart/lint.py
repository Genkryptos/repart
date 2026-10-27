"""Geometry sanity checks the silhouettes can't make.

A photo outline can hide mistakes: if nothing in the photo shows which way a feature
points, the optimiser may rotate the part until the mistake sits inside the outline.
These checks run on the mesh itself and go back to Spark as notes, not errors, since
some parts (clips, brackets) overhang on purpose.
"""

from __future__ import annotations

import cv2
import numpy as np
import trimesh


def _clock(dx: float, dy: float) -> int:
    """Model-frame direction as a clock position: +y is 12 o'clock, +x is 3 o'clock."""
    ang = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    h = int(round(ang / 30)) % 12
    return 12 if h == 0 else h


def overhang(mesh: trimesh.Trimesh, base_band_mm: float = 0.5) -> tuple[float, int]:
    """How far (mm) any part of the model reaches outside its own base footprint, and where."""
    v = mesh.vertices
    z0 = v[:, 2].min()
    base = v[v[:, 2] <= z0 + base_band_mm][:, :2]
    if len(base) < 3:
        return 0.0, 12
    hull = cv2.convexHull(base.astype(np.float32))
    d = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in v[:, :2]])
    i = int(np.argmin(d))  # most negative = furthest outside
    if d[i] >= 0:
        return 0.0, 12
    c = base.mean(axis=0)
    return float(-d[i]), _clock(v[i, 0] - c[0], v[i, 1] - c[1])


def notes(mesh: trimesh.Trimesh) -> list[str]:
    out = []
    dist, clock = overhang(mesh)
    if dist > 0.5:
        out.append(f"Part of the model overhangs its own base by {dist:.1f} mm toward {clock} o'clock "
                   "(model frame: +x = 3 o'clock, +y = 12 o'clock). The photos may not show this. "
                   "If this part shouldn't overhang (e.g. a ridge meant to stay inside the rim), fix the code.")
    if not mesh.is_watertight:
        out.append("The mesh isn't watertight, so slicers may fail; check for zero-thickness or touching faces.")
    return out
