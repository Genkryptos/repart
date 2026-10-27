"""OpenSCAD -> STL -> silhouette seen through a recovered phone camera."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .camera import Camera
from .config import CACHE_DIR

OPENSCAD = os.environ.get("OPENSCAD") or shutil.which("openscad") or "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"


class ScadError(RuntimeError):
    """OpenSCAD rejected the code; the message is fed back to the agent."""


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return repr(float(v)) if isinstance(v, float) else str(v)


def scad_to_mesh(code: str, params: dict | None = None, timeout: float = 60) -> tuple[trimesh.Trimesh, float]:
    """Compile SCAD (with -D parameter overrides) to a mesh. Cached by code + params.

    OpenSCAD code cannot touch the filesystem or network beyond include/import, so this
    is a much safer thing for an LLM to write than arbitrary Python.
    """
    params = params or {}
    defs = [f"{k}={_fmt(v)}" for k, v in sorted(params.items())]
    key = hashlib.sha256((code + "\n".join(defs)).encode()).hexdigest()[:20]
    out = CACHE_DIR / "stl" / f"{key}.stl"
    t0 = time.perf_counter()
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "part.scad"
            src.write_text(code)
            cmd = [OPENSCAD, "--backend", "manifold", "--export-format", "binstl", "-o", str(out)]
            for d in defs:
                cmd += ["-D", d]
            r = subprocess.run(cmd + [str(src)], capture_output=True, text=True, timeout=timeout)
            if r.returncode != 0 and "backend" in (r.stderr or "").lower():
                # Older OpenSCAD without Manifold: same output, just slower (CGAL).
                r = subprocess.run([c for c in cmd if c not in ("--backend", "manifold")] + [str(src)],
                                   capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not out.exists():
            raise ScadError((r.stderr or r.stdout).strip()[-1500:])
    mesh = trimesh.load_mesh(out, file_type="stl")
    if mesh.is_empty:
        raise ScadError("OpenSCAD produced an empty mesh (top-level object missing?).")
    return mesh, time.perf_counter() - t0


def silhouette(
    mesh: trimesh.Trimesh,
    cam: Camera,
    supersample: int = 4,
    pad: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Soft silhouette (0..1 coverage) of the mesh in the camera image, cropped.

    Returns (mask, (x0, y0, x1, y1)) where the mask covers image pixels [y0:y1, x0:x1].
    Every triangle is filled, so hidden faces don't matter for the outline.
    """
    v2 = cam.project(mesh.vertices)
    x0, y0 = np.floor(v2.min(axis=0)).astype(int) - pad
    x1, y1 = np.ceil(v2.max(axis=0)).astype(int) + pad
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, cam.width), min(y1, cam.height)
    s = supersample
    canvas = np.zeros(((y1 - y0) * s, (x1 - x0) * s), dtype=np.uint8)
    shift = 4  # cv2 sub-pixel bits
    tris = np.round((v2[mesh.faces] - [x0, y0]) * s * (1 << shift)).astype(np.int32)
    # One call per triangle: fillPoly on many polygons uses even-odd filling, so
    # overlapping faces would cancel out.
    for tri in tris:
        cv2.fillConvexPoly(canvas, tri, 255, lineType=cv2.LINE_8, shift=shift)
    soft = cv2.resize(canvas.astype(np.float32) / 255.0, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    return soft, (x0, y0, x1, y1)


def full_frame(soft: np.ndarray, box: tuple[int, int, int, int], width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.float32)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = soft
    return out
