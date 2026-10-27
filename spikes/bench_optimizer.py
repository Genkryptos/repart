"""Nelder-Mead vs CMA-ES (parallel compiles) on the two-photo L-bracket, true topology,
wrong starting numbers. Threshold masks with holes carved, so no API calls.

    uv run python spikes/bench_optimizer.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import synth
from make_eval_parts import VIEWS, XY
from repart.camera import sheet_corners, solve_camera
from repart.fit import Observation, fit
from repart.holes import carve_holes
from repart.render import scad_to_mesh

CODE = """$fn=64; L=40; W=20; T=3; H=20; WT=3; HD=5; HOLE_X=-4; PITCH=16;
difference(){
  union(){ translate([-L/2,-W/2,0]) cube([L,W,T]); translate([L/2-WT,-W/2,0]) cube([WT,W,H]); }
  for (s=[-1,1]) translate([HOLE_X + s*PITCH/2,0,-1]) cylinder(d=HD, h=T+2);
}"""
TRUE = {"L": 40, "W": 20, "T": 3, "H": 20, "WT": 3, "HD": 5, "HOLE_X": -4, "PITCH": 16}
START = {"L": 36, "W": 22, "T": 4, "H": 15, "WT": 4, "HD": 4, "HOLE_X": -2, "PITCH": 14}


def view(i: int) -> Observation:
    off, roll, dxy, yaw = VIEWS[i]
    xy = XY + dxy
    cam = synth.camera(xy, off, roll)
    img = synth.photo(scad_to_mesh(CODE)[0], cam, xy, yaw, np.random.default_rng(10 + i))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sheet = (g > 170).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sheet)
    sheet = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    est = solve_camera(sheet_corners(sheet), img.shape[1], img.shape[0])
    hull = np.zeros_like(sheet)
    cnts, _ = cv2.findContours(sheet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(hull, [cv2.convexHull(max(cnts, key=cv2.contourArea))], 1)
    dark = ((g < 140) & (hull > 0)).astype(np.uint8)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(dark)
    cv2.fillPoly(solid, [max(cnts, key=cv2.contourArea)], 1)  # SAM-style solid mask
    mask, _ = carve_holes(solid.astype(np.float32), img, sheet.astype(np.float32))
    return Observation(img, mask, est)


def main() -> None:
    views = [view(0), view(1)]
    tun = list(START)
    for method in ("nelder-mead", "cma"):
        t0 = time.perf_counter()
        r = fit(CODE, dict(START), tun, views, method=method)
        err = {k: r.params[k] - TRUE[k] for k in ("L", "W", "H", "HD")}
        print(f"{method:12s} IoU {r.iou:.4f}  {time.perf_counter() - t0:5.1f}s  {r.evals} evals  "
              + "  ".join(f"{k} {r.params[k]:.2f} ({e:+.2f})" for k, e in err.items()))


if __name__ == "__main__":
    main()
