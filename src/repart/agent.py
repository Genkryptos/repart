"""The RePart loop: Muse Spark owns topology, the optimiser owns the numbers.

Each round:
  1. Spark writes (or revises) parametric SCAD and says which params this view can see.
  2. The optimiser fits placement only (= how good Spark's own numbers are), then
     placement + tunable params (= the hybrid).
  3. Spark gets the fitted values, what the optimiser had to change, residual thickness
     by clock position *after* the fit, and an overlay. Residuals left after an optimal
     fit are structural, so Spark either changes the structure or declares done.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import ledger
from .compare import residual_overlay
from .config import SPARK_MODEL, client
from .fit import FitResult, Observation, fit, sector_residuals
from .render import ScadError, scad_to_mesh
from .scad import bake_params, declared

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "repart_draft",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "what you see, what you changed and why (brief)"},
                "params": {"type": "object", "additionalProperties": {"type": "number"},
                           "description": "every top-level parameter, mm"},
                "tunable": {"type": "array", "items": {"type": "string"},
                            "description": "params whose effect is visible in this photo's silhouette; the optimiser tunes only these"},
                "measured": {"type": "array", "items": {"type": "string"},
                             "description": "params set from the user's caliper/ruler measurements; locked, never tuned"},
                "scad": {"type": "string"},
                "questions": {"type": "array", "items": {"type": "string"},
                              "description": "hidden dimensions only the user can measure"},
                "done": {"type": "boolean", "description": "true when remaining residuals are alignment noise"},
            },
            "required": ["reasoning", "params", "tunable", "measured", "scad", "questions", "done"],
            "additionalProperties": False,
        },
    },
}

SYSTEM = """You are RePart's CAD agent. You reconstruct a replacement for a broken household
plastic part as parametric OpenSCAD, working with a numeric optimiser.

Division of labour:
- YOU decide structure: which primitives, holes, ridges, tapers, chamfers exist.
- The OPTIMISER decides numbers: it tunes the params you list in `tunable` plus the part's
  position, to maximise silhouette overlap with the photo through the calibrated camera.
- List as tunable only params that change the outline seen from this camera. Hidden features
  (bores, underside details) are not tunable; ask the user about them.
- A dimension the user measured with calipers or a ruler is ground truth: put it in `measured`,
  never in `tunable`. A single tilted view trades height against width, so a locked measurement
  makes every other fitted number more accurate.

Code rules:
- Declare every dimension as a top-level `name = value;` (mm), keep names stable across rounds.
- Part rests on z=0, centred on the origin in x/y. $fn >= 96 for round features.
- Declare `fit_gap = 0.2;` and add it to every hole, socket or slot that mates with another
  part (e.g. `d = shaft_d + 2 * fit_gap`). The loop sets it to 0 while matching the photo and
  to the material's clearance on export. Never list fit_gap as tunable or measured.
- Prefer the simplest structure that explains the outline. Don't add a taper, chamfer or skirt
  unless the residuals demand it: extra freedom the camera can't see makes the fit ambiguous.

Reading feedback (all residuals are measured AFTER the optimiser's best fit):
- "missing" = in the photo but not in your model; "extra" = in your model but not the photo.
- Thin (< 0.15 mm), scattered residuals are edge noise: set done=true.
- A thick residual concentrated at one clock position means a missing or wrong feature there.
- If the optimiser had to change a param a lot, your structure is probably wrong."""


def _img_url(bgr: np.ndarray, max_side: int = 1024) -> str:
    s = max_side / max(bgr.shape[:2])
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


@dataclass
class Round:
    index: int
    draft: dict
    compile_error: str | None = None
    spark_only: FitResult | None = None  # placement fitted, Spark's numbers kept
    hybrid: FitResult | None = None  # placement + tunable params fitted
    residuals: list[dict] = field(default_factory=list)
    spark_seconds: float = 0.0
    cost_usd: float = 0.0


def _feedback(r: Round, obs: Observation) -> list[dict]:
    if r.compile_error:
        return [{"type": "text", "text": f"OpenSCAD error, fix the code:\n{r.compile_error}"}]
    h, s = r.hybrid, r.spark_only
    changes = "\n".join(f"  {k}: {a:g} → {b:g} ({b - a:+.2f})" for k, (a, b) in h.changes.items()) or "  (none)"
    res = "\n".join(f"  {d['clock']} o'clock: {d['kind']} {d['thickness_mm']} mm" for d in r.residuals[:8]) or "  none ≥ 0.05 mm"
    text = (
        f"Your numbers as drafted (placement fitted): IoU {s.iou:.4f}.\n"
        f"After optimising your tunable params: IoU {h.iou:.4f}, mean edge error {h.comparison.chamfer_mm:.3f} mm, "
        f"95th percentile {h.comparison.hausdorff_mm:.3f} mm.\n"
        f"Optimiser changes:\n{changes}\n"
        f"Residuals after the fit, by clock position in the photo:\n{res}\n"
        "Overlay attached (green match, red missing, blue extra). Return the full draft: keep the "
        "optimised values as your new params, change structure only if residuals demand it, "
        "or set done=true."
    )
    overlay = residual_overlay(obs.image_crop, h.rendered, obs.mask_crop)
    return [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": _img_url(overlay)}}]


def run(
    obs: Observation,
    description: str,
    answers: list[str] | None = None,
    max_rounds: int = 4,
    out_dir: Path | None = None,
    reasoning_effort: str = "high",
    log=print,
) -> list[Round]:
    api = client()
    x0, y0, x1, y1 = obs.crop
    mm_px = obs.cam.mm_per_px_at([*obs.initial_xy(0), 0])
    intro = (
        f"{description}\n"
        f"Camera: {obs.cam.center[2]:.0f} mm above the sheet, tilted {obs.tilt_deg():.0f}° from vertical, "
        f"~{mm_px * 1000:.0f} µm per pixel at the sheet. The crop below is {x1 - x0}×{y1 - y0} px.\n"
        + (("User answers:\n" + "\n".join(f"- {a}" for a in answers) + "\n") if answers else "")
        + "Write the first parametric draft."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": intro},
                                     {"type": "image_url", "image_url": {"url": _img_url(obs.image_crop)}}]},
    ]
    rounds: list[Round] = []
    placement = None
    for i in range(max_rounds):
        ledger.ensure(0.05, f"Spark round {i}")
        t0 = time.perf_counter()
        resp = api.chat.completions.create(model=SPARK_MODEL, reasoning_effort=reasoning_effort,
                                           response_format=SCHEMA, messages=messages)
        cost = ledger.chat_cost(SPARK_MODEL, resp.usage)
        ledger.record(SPARK_MODEL, cost, stage="agent", round=i)
        raw = resp.choices[0].message.content
        draft = json.loads(raw)
        r = Round(i, draft, spark_seconds=time.perf_counter() - t0, cost_usd=cost)
        rounds.append(r)
        log(f"\n== round {i}: Spark {r.spark_seconds:.0f}s ${cost:.4f}  done={draft['done']}  "
            f"tunable={draft['tunable']}  measured={draft['measured']}")
        log("   " + draft["reasoning"][:260].replace("\n", " "))

        if i > 0 and draft["done"]:
            log("   Spark declares done.")
            break
        params = {k: float(v) for k, v in draft["params"].items() if k in declared(draft["scad"])}
        if "fit_gap" in declared(draft["scad"]):
            params["fit_gap"] = 0.0  # match true geometry; clearance is added on export
        tunable = [k for k in draft["tunable"] if k not in set(draft["measured"]) and k != "fit_gap"]
        try:
            scad_to_mesh(draft["scad"], params)
            r.spark_only = fit(draft["scad"], params, [], obs, placement=placement)
            r.hybrid = fit(draft["scad"], params, tunable, obs, placement=r.spark_only.placement)
        except ScadError as e:
            r.compile_error = str(e)[-800:]
            log(f"   compile error: {r.compile_error[:160]}")
        else:
            placement = r.hybrid.placement
            mm = obs.cam.mm_per_px_at([*placement[:2], 0])
            r.residuals = sector_residuals(r.hybrid.rendered, obs.mask_crop, mm)
            log(f"   Spark's numbers IoU {r.spark_only.iou:.4f} → hybrid IoU {r.hybrid.iou:.4f}  "
                f"edge {r.hybrid.comparison.chamfer_mm * 1000:.0f} µm  ({r.hybrid.evals} evals, {r.hybrid.seconds:.0f}s)")
            log("   changes: " + ", ".join(f"{k} {a:g}→{b:g}" for k, (a, b) in r.hybrid.changes.items()))
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"round{i}.scad").write_text(bake_params(draft["scad"], r.hybrid.params))
                cv2.imwrite(str(out_dir / f"round{i}_overlay.jpg"),
                            residual_overlay(obs.image_crop, r.hybrid.rendered, obs.mask_crop))
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": _feedback(r, obs)}]
    return rounds


def best_round(rounds: list[Round]) -> Round | None:
    fitted = [r for r in rounds if r.hybrid]
    return max(fitted, key=lambda r: r.hybrid.iou) if fitted else None
