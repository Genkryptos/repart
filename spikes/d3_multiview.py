"""Day-4 test: does a second, steep photo fix the height?

Photo A: the usual ~20° tilt. Photo B: ~48° tilt from the other side, with the knob
moved and rotated on the sheet, so placement must be fitted per photo. Fits the true
knob SCAD (D, H tunable, started wrong at 35 / 15 mm) from A, B, and A+B, using
threshold masks. With --spark, also runs the full agent on both photos.

    uv run python spikes/d3_multiview.py
    uv run python spikes/d3_multiview.py --spark
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import d1_parallax as d1
from d2_hybrid import TRUTH, mesh_dims
from repart import agent, ledger
from repart.camera import Camera, focal_from_35mm, intrinsics, sheet_corners, solve_camera
from repart.config import RUNS_DIR
from repart.fit import Observation, fit

VIEWS = {  # name: (camera offset from part mm, roll deg, part xy on sheet, part yaw deg)
    "A 20°": ([60, -95, 310], -6, d1.PART_XY, 35),
    "B 48°": ([-200, 160, 230], 8, d1.PART_XY + np.array([-12.0, 9.0]), 80),
}


def make_view(offset, roll, xy, yaw_deg, seed) -> Observation:
    f = focal_from_35mm(d1.F35, d1.W_IMG, d1.H_IMG)
    target = np.array([*xy, 0.0])
    rvec, tvec = d1.look_at(target + offset, target, roll)
    true_cam = Camera(intrinsics(d1.W_IMG, d1.H_IMG, f), rvec, tvec, d1.W_IMG, d1.H_IMG)
    img, _ = d1.synth_photo(true_cam, np.random.default_rng(seed), xy=xy, yaw=np.radians(yaw_deg))
    sheet, part = d1.masks_from_photo(img)
    return Observation(img, part, solve_camera(sheet_corners(sheet), d1.W_IMG, d1.H_IMG))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", action="store_true", help="also run the agent on both photos")
    args = ap.parse_args()

    views = {}
    for i, (name, (off, roll, xy, yaw)) in enumerate(VIEWS.items()):
        views[name] = make_view(np.array(off, float), roll, xy, yaw, seed=i)
        o = views[name]
        print(f"{name}: recovered tilt {o.tilt_deg():.1f}°, camera {o.cam.center[2]:.0f} mm up, "
              f"focal {o.cam.focal:.0f} px (true {focal_from_35mm(d1.F35, d1.W_IMG, d1.H_IMG):.0f})")

    start = {"D": 35.0, "H": 15.0, "ridge_w": 3.0, "ridge_h": 2.0}
    print("\nTrue knob SCAD, D and H tunable, started at D=35, H=15 (truth 32 / 18):")
    for label, vs in [("A only", [views["A 20°"]]), ("B only", [views["B 48°"]]),
                      ("A + B", list(views.values()))]:
        t0 = time.perf_counter()
        r = fit(d1.KNOB, start, ["D", "H"], vs)
        print(f"  {label:7s} D {r.params['D']:.3f} ({r.params['D'] - 32:+.3f})  H {r.params['H']:.3f} "
              f"({r.params['H'] - 18:+.3f})  IoU {' / '.join(f'{v:.4f}' for v in r.ious)}  "
              f"{r.evals} evals {time.perf_counter() - t0:.0f}s")

    if args.spark:
        out = RUNS_DIR / "d3"
        spent0 = ledger.spent()
        rounds = agent.run(list(views.values()), "Two photos of a fan-regulator knob on an A4 sheet; "
                           "it was moved between shots.",
                           answers=["There is a raised pointer ridge on top: 2 mm tall, 3 mm wide, running "
                                    "from near the centre to about 3 mm from the edge."],
                           out_dir=out)
        best = agent.best_round(rounds)
        dims = mesh_dims((out / f"round{best.index}.scad").read_text())
        print("\nAgent on A + B (no height given):")
        print("  " + "  ".join(f"{k} {dims[k]:.2f} ({dims[k] - v:+.2f})" for k, v in TRUTH.items()))
        print(f"  cost ${ledger.spent() - spent0:.4f}")


if __name__ == "__main__":
    main()
