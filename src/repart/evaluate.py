"""Score generated parts against caliper ground truth.

Each part lives in data/parts/<id>/ with its photo and a part.json:

    {
      "photos": ["top_tilt.jpg", "steep.jpg"],
      "concept": "black plastic knob",
      "description": "Fan regulator knob; D-shaped hole fits the shaft.",
      "answers": ["Shaft hole is 6 mm with a flat, 10 mm deep."],
      "material": "PETG",
      "calipers": [
        {"name": "outer diameter", "kind": "diameter_at", "z": 5, "mm": 32.1},
        {"name": "height", "kind": "height", "mm": 18.0}
      ]
    }

`answers` are hidden dimensions a user would type in. Don't repeat a scored caliper
value there, or that dimension is trivially exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .camera import CaptureError
from .fit import centre_mesh
from .pipeline import PartRequest, make_part

KINDS = ("height", "length", "width", "diameter_at", "hole_diameter_at")


def footprint(points_xy: np.ndarray) -> tuple[float, float]:
    """(long, short) sides of the minimum-area rectangle around the points."""
    (_, _), (a, b), _ = cv2.minAreaRect(points_xy.astype(np.float32))
    return max(a, b), min(a, b)


def measure(mesh: trimesh.Trimesh, kind: str, z: float | None = None) -> float:
    m = centre_mesh(mesh)
    if kind == "height":
        return float(m.bounds[1][2])
    if kind in ("length", "width"):
        long, short = footprint(m.vertices[:, :2])
        return float(long if kind == "length" else short)
    if kind == "diameter_at":
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            return float("nan")
        return float(footprint(sec.vertices[:, :2])[0])
    if kind == "hole_diameter_at":
        # Every closed loop of the cross-section; the outline is the biggest, the rest are
        # holes. Report the biggest hole (nan if the section has none).
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            return float("nan")
        loops = sorted((footprint(np.asarray(d)[:, :2]) for d in sec.discrete), key=lambda lw: -lw[0] * lw[1])
        return float(loops[1][0]) if len(loops) > 1 else float("nan")
    raise ValueError(f"Unknown measurement kind {kind!r}; use one of {KINDS}.")


def evaluate(parts_dir: Path, runs_dir: Path, rerun: bool = False, log=print) -> list[dict]:
    rows = []
    for spec_path in sorted(parts_dir.glob("*/part.json")):
        pid = spec_path.parent.name
        spec = json.loads(spec_path.read_text())
        for c in spec["calipers"]:
            if any(f"{c['mm']:g}" in a for a in spec.get("answers", [])):
                log(f"warning: {pid}: scored value {c['mm']} for '{c['name']}' also appears in answers")
        out = runs_dir / pid
        if rerun or not (out / "report.json").exists():
            log(f"\n### {pid}")
            try:
                report = make_part(PartRequest(
                    photos=[spec_path.parent / p for p in spec.get("photos") or [spec["photo"]]],
                    concept=spec["concept"], out_dir=out,
                    description=spec.get("description", ""), answers=spec.get("answers", []),
                    material=spec.get("material", "PETG"), fit_gap=0.0,  # score true geometry
                    focal35=spec.get("focal35"),
                ), log=log)
            except (CaptureError, RuntimeError) as exc:  # one bad part shouldn't stop the run
                log(f"FAILED {pid}: {exc}")
                rows += [{"part": pid, "dimension": c["name"], "caliper_mm": c["mm"], "model_mm": float("nan"),
                          "error_mm": float("nan"), "iou": float("nan"), "cost_usd": 0.0, "failed": str(exc)}
                         for c in spec["calipers"]]
                continue
        else:
            report = json.loads((out / "report.json").read_text())
        mesh = trimesh.load_mesh(out / "part.stl")
        for c in spec["calipers"]:
            got = measure(mesh, c["kind"], c.get("z"))
            rows.append({"part": pid, "dimension": c["name"], "caliper_mm": c["mm"], "model_mm": got,
                         "error_mm": got - c["mm"], "iou": report["iou"], "cost_usd": report["cost_usd"]})
    return rows


def summary(rows: list[dict]) -> str:
    if not rows:
        return "No parts found."
    lines = ["| part | dimension | caliper | model | error |", "|---|---|---|---|---|"]
    lines += [f"| {r['part']} | {r['dimension']} | {r['caliper_mm']:.2f} | {r['model_mm']:.2f} | {r['error_mm']:+.2f} |"
              for r in rows]
    failed = sorted({r["part"] for r in rows if r.get("failed")})
    scored = [r for r in rows if not r.get("failed")]
    missing = [r for r in scored if not np.isfinite(r["model_mm"])]  # e.g. a hole the model lacks
    errs = np.abs([r["error_mm"] for r in scored if np.isfinite(r["model_mm"])])
    if missing:
        lines.append("")
        lines.append("missing from the model: " + ", ".join(f"{r['part']} {r['dimension']}" for r in missing))
    if not len(errs):
        return "\n".join(lines + ["", f"All parts failed: {', '.join(failed)}"])
    parts = {r["part"]: r for r in rows}
    total = len(errs) + len(missing)  # a missing feature counts as a miss
    lines += ["", f"median |error| {np.median(errs):.2f} mm, max {errs.max():.2f} mm, "
                  f"{(errs <= 0.5).sum() / total:.0%} of dimensions within 0.5 mm, "
                  f"{(errs <= 1.0).sum() / total:.0%} within 1 mm; "
                  f"{len(parts) - len(failed)} parts, ${sum(r['cost_usd'] for r in parts.values()):.2f}"]
    if failed:
        lines.append(f"failed (counted separately): {', '.join(failed)}")
    return "\n".join(lines)
