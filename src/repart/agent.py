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
from . import lint
from .compare import residual_overlay
from .config import SPARK_MODEL, WONDER_MODEL, client
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
                "through_holes": {"type": "boolean",
                                  "description": "true if the reported through-hole candidates are real holes; false if they are printing, labels or glare"},
            },
            "required": ["reasoning", "params", "tunable", "measured", "scad", "questions", "done", "through_holes"],
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
- List as tunable only params that change the outline seen from these cameras. Hidden features
  (bores, underside details) are not tunable; ask the user about them.
- There may be several photos of the same part, possibly moved between shots. A steep photo
  shows the side wall and pins down heights; a near-vertical one pins down widths.
- A dimension the user measured with calipers or a ruler is ground truth: put it in `measured`,
  never in `tunable`. A single tilted view trades height against width, so a locked measurement
  makes every other fitted number more accurate.

Use the user's description to choose the overall shape: "L-shaped" means a wall standing up from
a base, "C-shaped" an open ring, and so on. A large "missing" region in a steep photo usually
means a vertical feature (wall, boss, tab) your model doesn't have yet; add it rather than
stretching the numbers.

Code rules:
- Declare every dimension as a top-level `name = value;` (mm), keep names stable across rounds.
- Name parameters that *place* a feature with `_x`, `_y`, `offset` or `cx`/`cy` (these may go
  negative); every other parameter is a size and is kept positive by the optimiser.
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
- If the optimiser had to change a param a lot, your structure is probably wrong.
- Geometry-check notes describe the mesh itself, independent of the photos. Photos can hide a
  mistake (e.g. the optimiser rotated the part so an error sits inside the outline), so act on
  them before setting done=true unless the feature is meant to be that way."""


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
    residuals: list[list[dict]] = field(default_factory=list)  # per view
    lint: list[str] = field(default_factory=list)
    spark_seconds: float = 0.0
    model: str = SPARK_MODEL
    views: list = field(default_factory=list)  # the observations this round was fitted against
    cost_usd: float = 0.0


def _feedback(r: Round, views: list[Observation]) -> list[dict]:
    if r.compile_error:
        return [{"type": "text", "text": f"OpenSCAD error, fix the code:\n{r.compile_error}"}]
    h, s = r.hybrid, r.spark_only
    changes = "\n".join(f"  {k}: {a:g} → {b:g} ({b - a:+.2f})" for k, (a, b) in h.changes.items()) or "  (none)"
    lines = [
        f"Your numbers as drafted (placement fitted): mean IoU {s.iou:.4f}.",
        f"After optimising your tunable params: mean IoU {h.iou:.4f}, mean edge error {h.chamfer_mm:.3f} mm.",
        f"Optimiser changes:\n{changes}",
    ]
    content: list[dict] = []
    for j, (obs, res) in enumerate(zip(views, r.residuals)):
        rs = "\n".join(f"    {d['clock']} o'clock: {d['kind']} {d['thickness_mm']} mm" for d in res[:8]) or "    none ≥ 0.05 mm"
        lines.append(f"Photo {j + 1} (tilt {obs.tilt_deg():.0f}°): IoU {h.ious[j]:.4f}, "
                     f"95th pct edge {h.comparisons[j].hausdorff_mm:.3f} mm. Residuals after the fit:\n{rs}")
    if r.lint:
        lines.append("Geometry check on your model:\n" + "\n".join(f"  - {n}" for n in r.lint))
    lines.append("Overlays attached in photo order (green match, red missing, blue extra). Return the full "
                 "draft: keep the optimised values as your new params, change structure only if residuals "
                 "demand it, or set done=true.")
    content.append({"type": "text", "text": "\n".join(lines)})
    for obs, rend in zip(views, h.rendered):
        content.append({"type": "image_url", "image_url": {"url": _img_url(residual_overlay(obs.image_crop, rend, obs.mask_crop))}})
    return content


def run(
    views: Observation | list[Observation],
    description: str,
    answers: list[str] | None = None,
    max_rounds: int = 4,
    out_dir: Path | None = None,
    reasoning_effort: str = "high",
    log=print,
    on_round=None,
) -> list[Round]:
    """on_round(round) is called after each round is fitted (for live progress UIs)."""
    views = [views] if isinstance(views, Observation) else list(views)
    api = client()
    cams = []
    for j, obs in enumerate(views):
        x0, y0, x1, y1 = obs.crop
        mm_px = obs.cam.mm_per_px_at([*obs.initial_xy(0), 0])
        cams.append(f"Photo {j + 1}: camera {obs.cam.center[2]:.0f} mm above the sheet, tilted "
                    f"{obs.tilt_deg():.0f}° from vertical, ~{mm_px * 1000:.0f} µm/px; crop {x1 - x0}×{y1 - y0} px.")
    for j, obs in enumerate(views):
        if obs.hole_areas_mm2:
            sizes = ", ".join(f"~{np.sqrt(4 * a / np.pi):.1f} mm" for a in sorted(obs.hole_areas_mm2, reverse=True)[:6])
            cams.append(f"Photo {j + 1}: {len(obs.hole_areas_mm2)} enclosed region(s) inside the part look like the "
                        f"sheet showing through (equivalent diameters {sizes}). They may be real through-holes, or "
                        "white printing, labels or glare; say which with `through_holes`.")
    intro = (
        f"{description}\n" + "\n".join(cams) + "\n"
        + (("User answers:\n" + "\n".join(f"- {a}" for a in answers) + "\n") if answers else "")
        + "Write the first parametric draft."
    )
    first = [{"type": "text", "text": intro}] + [
        {"type": "image_url", "image_url": {"url": _img_url(obs.image_crop)}} for obs in views]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": first}]
    rounds: list[Round] = []
    placements = None
    for i in range(max_rounds):
        model = WONDER_MODEL if _stalled(rounds) else SPARK_MODEL
        if model != SPARK_MODEL:
            log(f"   fit has stalled below IoU {STALL_IOU}; escalating this round to {model}")
        ledger.ensure(0.05 if model == SPARK_MODEL else 0.40, f"{model} round {i}")
        t0 = time.perf_counter()
        resp = api.chat.completions.create(model=model, reasoning_effort=reasoning_effort,
                                           response_format=SCHEMA, messages=messages)
        cost = ledger.chat_cost(model, resp.usage)
        ledger.record(model, cost, stage="agent", round=i)
        raw = resp.choices[0].message.content
        draft = json.loads(raw)
        r = Round(i, draft, spark_seconds=time.perf_counter() - t0, cost_usd=cost, model=model)
        rounds.append(r)
        log(f"\n== round {i}: {model} {r.spark_seconds:.0f}s ${cost:.4f}  done={draft['done']}  "
            f"tunable={draft['tunable']}  measured={draft['measured']}")
        log("   " + draft["reasoning"][:260].replace("\n", " "))

        if i > 0 and draft["done"]:
            log("   Spark declares done.")
            if on_round:
                on_round(r)
            break
        params = {k: float(v) for k, v in draft["params"].items() if k in declared(draft["scad"])}
        if "fit_gap" in declared(draft["scad"]):
            params["fit_gap"] = 0.0  # match true geometry; clearance is added on export
        tunable = [k for k in draft["tunable"] if k not in set(draft["measured"]) and k != "fit_gap"]
        fit_views = views if draft.get("through_holes", True) else [v.without_holes() for v in views]
        if fit_views is not views:
            log("   Spark says the hole candidates aren't real holes; fitting the uncarved masks")
        try:
            scad_to_mesh(draft["scad"], params)
            r.spark_only = fit(draft["scad"], params, [], fit_views, placements=placements)
            r.hybrid = fit(draft["scad"], params, tunable, fit_views, placements=r.spark_only.placements)
        except ScadError as e:
            r.compile_error = str(e)[-800:]
            log(f"   compile error: {r.compile_error[:160]}")
        else:
            placements = r.hybrid.placements
            r.lint = lint.notes(scad_to_mesh(draft["scad"], r.hybrid.params)[0])
            for note in r.lint:
                log(f"   geometry check: {note[:120]}")
            r.residuals = [sector_residuals(rend, obs.mask_crop, obs.cam.mm_per_px_at([*pl[:2], 0]))
                           for rend, obs, pl in zip(r.hybrid.rendered, fit_views, placements)]
            per_view = " / ".join(f"{v:.4f}" for v in r.hybrid.ious)
            log(f"   Spark's numbers IoU {r.spark_only.iou:.4f} → hybrid IoU {r.hybrid.iou:.4f} [{per_view}]  "
                f"edge {r.hybrid.chamfer_mm * 1000:.0f} µm  ({r.hybrid.evals} evals, {r.hybrid.seconds:.0f}s)")
            log("   changes: " + ", ".join(f"{k} {a:g}→{b:g}" for k, (a, b) in r.hybrid.changes.items()))
            if out_dir:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"round{i}.scad").write_text(bake_params(draft["scad"], r.hybrid.params))
                for j, (obs, rend) in enumerate(zip(fit_views, r.hybrid.rendered)):
                    cv2.imwrite(str(out_dir / f"round{i}_view{j + 1}_overlay.jpg"),
                                residual_overlay(obs.image_crop, rend, obs.mask_crop))
        if on_round:
            on_round(r)
        r.views = fit_views
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": _feedback(r, fit_views)}]
    return rounds


STALL_IOU = 0.97


def _stalled(rounds: list[Round]) -> bool:
    """Two fitted rounds in a row below STALL_IOU with < 0.01 gain: the structure is
    probably wrong, which is Muse Wonder's job rather than another Spark nudge."""
    fitted = [r.hybrid.iou for r in rounds if r.hybrid]
    return len(fitted) >= 2 and fitted[-1] < STALL_IOU and fitted[-1] - fitted[-2] < 0.01


def best_round(rounds: list[Round]) -> Round | None:
    fitted = [r for r in rounds if r.hybrid]
    return max(fitted, key=lambda r: r.hybrid.iou) if fitted else None
