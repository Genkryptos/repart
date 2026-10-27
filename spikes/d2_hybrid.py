"""Day-2 test: the hybrid loop (Spark topology + optimiser numbers) on the synthetic knob.

Truth: straight cylinder D = 32 mm, H = 18 mm, plus a 2 mm pointer ridge on top
(total height 20 mm). Scores the final mesh by measuring its diameter at three
heights (catches a wrong taper) and its total height, and prints the ablation per
round: Spark's own numbers vs. the optimised ones.

    uv run python spikes/d2_hybrid.py                       # photo only
    uv run python spikes/d2_hybrid.py --answer "Calipers: 18 mm from base to the flat top; a 2 mm pointer ridge sits on top"
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import d1_parallax as d1
from repart import agent, ledger
from repart.camera import Camera, focal_from_35mm, intrinsics, sheet_corners, solve_camera
from repart.config import RUNS_DIR
from repart.fit import Observation, centre_mesh
from repart.render import scad_to_mesh

TRUTH = {"d@3": 32.0, "d@9": 32.0, "d@15": 32.0, "height": 20.0}


def mesh_dims(code: str) -> dict:
    m = centre_mesh(scad_to_mesh(code)[0])
    out = {}
    for z in (3, 9, 15):
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        out[f"d@{z}"] = float(max(np.ptp(sec.vertices[:, 0]), np.ptp(sec.vertices[:, 1]))) if sec else 0.0
    out["height"] = float(m.bounds[1][2])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", action="append", help="simulated user answer (repeatable)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tag", default="photo_only")
    args = ap.parse_args()
    out = RUNS_DIR / "d2" / args.tag

    f = focal_from_35mm(d1.F35, d1.W_IMG, d1.H_IMG)
    t = np.array([*d1.PART_XY, 0.0])
    rvec, tvec = d1.look_at(t + [60, -95, 310], t, -6)
    img, _ = d1.synth_photo(Camera(intrinsics(d1.W_IMG, d1.H_IMG, f), rvec, tvec, d1.W_IMG, d1.H_IMG),
                            np.random.default_rng(0))
    sheet, part = d1.masks_from_photo(img)
    obs = Observation(img, part, solve_camera(sheet_corners(sheet), d1.W_IMG, d1.H_IMG))

    spent0 = ledger.spent()
    rounds = agent.run(obs, "Photo of a fan-regulator knob lying on an A4 sheet (cropped).",
                       answers=args.answer, max_rounds=args.rounds, out_dir=out)

    print("\nround | Spark-only IoU | hybrid IoU | edge µm | " + " | ".join(TRUTH))
    table = []
    for r in rounds:
        if not r.hybrid:
            continue
        dims = mesh_dims((out / f"round{r.index}.scad").read_text())
        err = {k: dims[k] - v for k, v in TRUTH.items()}
        table.append({"round": r.index, "spark_iou": r.spark_only.iou, "hybrid_iou": r.hybrid.iou,
                      "edge_um": r.hybrid.chamfer_mm * 1000, "dims": dims, "err": err,
                      "tunable": r.draft["tunable"], "questions": r.draft["questions"]})
        print(f"  {r.index}   |     {r.spark_only.iou:.4f}     |   {r.hybrid.iou:.4f}   | {table[-1]['edge_um']:6.0f}  | "
              + " | ".join(f"{dims[k]:.2f} ({err[k]:+.2f})" for k in TRUTH))
    best = agent.best_round(rounds)
    if best:
        print(f"\nbest round {best.index}: max |error| "
              f"{max(abs(e) for e in next(t['err'] for t in table if t['round'] == best.index).values()):.2f} mm")
        print("questions Spark asked:", best.draft["questions"])
    print(f"cost ${ledger.spent() - spent0:.4f}, total spent ${ledger.spent():.3f}")
    (out / "results.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
