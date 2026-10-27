"""repart make PHOTO --part "black plastic knob"   |   repart eval data/parts"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import trimesh

from . import ledger
from .camera import CaptureError
from .config import RUNS_DIR
from .evaluate import evaluate, footprint, summary
from .pipeline import FIT_GAPS, PartRequest, make_part


def _make(args) -> int:
    out = args.out or RUNS_DIR / "parts" / args.photo.stem
    try:
        report = make_part(PartRequest(
            photo=args.photo, concept=args.part, out_dir=out, description=args.describe or "",
            answers=args.answer or [], material=args.material, fit_gap=args.fit_gap,
            max_rounds=args.rounds, focal35=args.focal35, sheet_concept=args.sheet,
        ))
    except CaptureError as e:
        print(f"\nRetake the photo: {e}", file=sys.stderr)
        return 2
    except ledger.SpendCapError as e:
        print(f"\nStopped by the spend cap: {e}", file=sys.stderr)
        return 3
    mesh = trimesh.load_mesh(out / "part.stl")
    long, short = footprint(mesh.vertices[:, :2])
    print(f"\nPart: {long:.2f} × {short:.2f} mm footprint, {mesh.bounds[1][2] - mesh.bounds[0][2]:.2f} mm tall")
    print(f"Fit: IoU {report['iou']:.4f}, mean edge error {report['edge_mm']:.3f} mm "
          f"(round {report['best_round']} of {len(report['rounds'])}), cost ${report['cost_usd']:.3f}")
    if report["fit_gap_mm"] is not None:
        print(f"Mating holes include a {report['fit_gap_mm']:.2f} mm fit gap for {report['material']}.")
    print(f"Files: {out / 'part.stl'}, {out / 'part.scad'}, {out / 'fit_overlay.jpg'}")
    if report["questions"]:
        print("\nRePart can't see these from the photo. Measure and re-run with --answer:")
        for q in report["questions"]:
            print(f"  - {q}")
    return 0


def _eval(args) -> int:
    rows = evaluate(args.parts, RUNS_DIR / "eval", rerun=args.rerun)
    text = summary(rows)
    print("\n" + text)
    (RUNS_DIR / "eval").mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "eval" / "summary.md").write_text(text + "\n")
    (RUNS_DIR / "eval" / "rows.json").write_text(json.dumps(rows, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="repart", description="Photo of a broken part → printable replacement.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="photo → STL")
    mk.add_argument("photo", type=Path)
    mk.add_argument("--part", required=True, help='short noun phrase SAM can find, e.g. "black plastic knob"')
    mk.add_argument("--describe", help="what the part does and anything notable")
    mk.add_argument("--answer", action="append", help="a measurement the photo can't show (repeatable)")
    mk.add_argument("--material", default="PETG", choices=sorted(FIT_GAPS))
    mk.add_argument("--fit-gap", type=float, help="radial clearance for mating holes, mm (overrides material)")
    mk.add_argument("--rounds", type=int, default=4)
    mk.add_argument("--focal35", type=float, help="35 mm-equivalent focal length if the photo lacks EXIF")
    mk.add_argument("--sheet", default="white paper sheet", help="SAM phrase for the reference sheet")
    mk.add_argument("--out", type=Path)
    mk.set_defaults(func=_make)

    ev = sub.add_parser("eval", help="score parts against caliper readings")
    ev.add_argument("parts", type=Path, nargs="?", default=Path("data/parts"))
    ev.add_argument("--rerun", action="store_true")
    ev.set_defaults(func=_eval)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
