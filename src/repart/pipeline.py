"""Photos in, printable part out: the one path the CLI and the web app both call."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import agent, ledger
from .camera import CaptureError, focal_from_35mm, sheet_corners, solve_camera
from .compare import residual_overlay
from .fit import Observation
from .holes import carve_holes
from .photo import Photo, load_photo
from .render import scad_to_mesh
from .sam import ImageMask, segment_photo
from .scad import bake_params, declared

# Radial clearance (mm) added to mating holes/sockets per material. FDM rules of thumb;
# calibrate on the first test print and override with fit_gap.
FIT_GAPS = {"PLA": 0.15, "PETG": 0.20, "ABS": 0.20, "Nylon": 0.25, "TPU": 0.10}


@dataclass
class PartRequest:
    photos: list[Path]  # one or more shots of the same part on the sheet; it may move between them
    concept: str  # SAM noun phrase for the part, e.g. "black plastic knob"
    out_dir: Path
    description: str = ""
    answers: list[str] = field(default_factory=list)
    sheet_concept: str = "white paper sheet"
    material: str = "PETG"
    fit_gap: float | None = None
    max_rounds: int = 5  # Spark usually finishes in 2; a stalled fit escalates to Wonder
    focal35: float | None = None  # override when photos have no EXIF


def _inside(poly: np.ndarray, pt: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32), pt, False) >= 0


SHEET_FALLBACKS = ["white paper", "sheet of paper"]
SHAPE_WORDS = {"round", "circular", "square", "rectangular", "flat", "small", "big", "large", "tiny",
               "long", "short", "thin", "thick", "broken", "old", "cracked"}


COLOURS = {"black", "white", "grey", "gray", "red", "blue", "green", "yellow", "brown", "orange",
           "beige", "cream", "pink", "purple", "transparent", "clear", "silver"}


def concept_fallbacks(concept: str) -> list[str]:
    """The user's phrase, then without shape words (a round knob isn't round in a steep shot),
    then the bare noun, then "<colour> object". SAM is cheap; a failed capture is not."""
    words = concept.split()
    colour = next((w for w in words if w.lower() in COLOURS), None)
    out = [concept, " ".join(w for w in words if w.lower() not in SHAPE_WORDS), words[-1]]
    if colour:
        out.append(f"{colour} object")
    return [c for i, c in enumerate(out) if c and c not in out[:i]]


def _pick_part(masks: list[ImageMask], corners: np.ndarray, concept: str, which: str) -> ImageMask:
    on_sheet = [m for m in masks if m.area > 0 and _inside(corners, m.centroid())]
    if not on_sheet:
        raise CaptureError(f"{which}: couldn't find a '{concept}' on the sheet. Try a different description, "
                           "or a background that contrasts with the part.")
    return max(on_sheet, key=lambda m: m.area)


def _sheet_point(cam, part: ImageMask) -> np.ndarray:
    """Where the part's mask centroid hits the sheet plane, mm (for a local mm-per-pixel)."""
    x, y = part.centroid()
    p = cam.plane_homography() @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def observe(photo: Photo, req: PartRequest, which: str, log=print) -> tuple[Observation, dict]:
    """SAM masks + camera for one photo, with retake messages for anything unmeasurable."""
    sheets, parts = asyncio.run(_segment(photo, req))
    # SAM's text grounding is phrase-sensitive (one photo found nothing for "white paper sheet"
    # but the full sheet for "white paper"), so the sheet gets fallbacks too.
    for alt in SHEET_FALLBACKS:
        if sheets:
            break
        if alt != req.sheet_concept:
            log(f"{which}: no sheet for '{req.sheet_concept}', trying '{alt}'")
            sheets = asyncio.run(segment_photo(photo.path, alt, photo.width, photo.height))
    tried = req.concept
    for alt in concept_fallbacks(req.concept)[1:]:
        if parts:
            break
        log(f"{which}: nothing found for '{tried}', trying '{alt}'")
        parts = asyncio.run(segment_photo(photo.path, alt, photo.width, photo.height))
        tried = alt
    if not sheets:
        raise CaptureError(f"{which}: couldn't find the A4 sheet. Use plain white paper on a darker surface.")
    sheet = max(sheets, key=lambda m: m.area)
    try:
        corners = sheet_corners(sheet.mask > 0.5)
    except CaptureError as e:
        raise CaptureError(f"{which}: {e}") from e
    f35 = req.focal35 or photo.focal35
    focal_px = focal_from_35mm(f35, photo.width, photo.height) if f35 else None
    try:
        cam = solve_camera(corners, photo.width, photo.height, focal_px=focal_px)
    except ValueError as e:
        raise CaptureError(f"{which}: can't work out the camera. Tilt the phone 15-20° "
                           "instead of shooting straight down.") from e
    part = _pick_part(parts, corners, req.concept, which)
    # SAM masks are solid; through-holes show the sheet, so carve those back out.
    mask, holes = carve_holes(part.mask, photo.image, sheet.mask)
    mm_px = cam.mm_per_px_at([*_sheet_point(cam, part), 0])
    obs = Observation(photo.image, mask, cam, solid_mask=part.mask if holes.holes else None,
                      hole_areas_mm2=[a * mm_px**2 for a in holes.areas_px])
    focal_src = "given" if req.focal35 else "EXIF" if photo.focal35 else "estimated"
    log(f"{which}: {photo.width}×{photo.height}, focal {focal_src}{f' {f35:g} mm-equiv' if f35 else ''}; "
        f"corner fit {cam.reproj_px:.2f} px, camera {cam.center[2]:.0f} mm up, tilt {obs.tilt_deg():.0f}°; "
        f"part {part.area:.0f} px, SAM downsample {part.downsample:.2f}×; "
        f"{holes.holes} through-hole(s) carved{' (' + holes.skipped + ')' if holes.skipped else ''}")
    info = {"height_mm": float(cam.center[2]), "tilt_deg": obs.tilt_deg(), "focal_px": cam.focal,
            "focal_source": focal_src, "corner_reproj_px": cam.reproj_px,
            "sam_part_downsample": part.downsample, "sam_part_confidence": part.confidence,
            "sam_part_masks": len(parts), "holes_carved": holes.holes}
    return obs, info


def make_part(req: PartRequest, log=print, on_view=None, on_round=None) -> dict:
    """on_view(index, info) after each photo is measured; on_round(round) after each agent round."""
    out = req.out_dir
    out.mkdir(parents=True, exist_ok=True)
    spent0 = ledger.spent()
    views, infos = [], []
    for j, src in enumerate(req.photos):
        obs, info = observe(load_photo(src, out / f"view{j + 1}"), req, f"photo {j + 1}", log)
        views.append(obs)
        infos.append(info)
        if on_view:
            on_view(j, info)

    rounds = agent.run(views, req.description or f"Photos of a broken household part ({req.concept}) on an A4 sheet.",
                       answers=req.answers, max_rounds=req.max_rounds, out_dir=out / "rounds", log=log,
                       on_round=on_round)
    best = agent.best_round(rounds)
    if best is None:
        raise RuntimeError("No draft compiled; see rounds/ for Spark's code and errors.")

    scad = best.draft["scad"]
    gap = req.fit_gap if req.fit_gap is not None else FIT_GAPS.get(req.material, 0.2)
    final_params = dict(best.hybrid.params)
    if "fit_gap" in declared(scad):
        final_params["fit_gap"] = gap
    code = bake_params(scad, final_params)
    (out / "part.scad").write_text(code)
    mesh, _ = scad_to_mesh(code)
    mesh.export(out / "part.stl")
    for j, (obs, rend) in enumerate(zip(best.views or views, best.hybrid.rendered)):
        cv2.imwrite(str(out / f"view{j + 1}" / "fit_overlay.jpg"), residual_overlay(obs.image_crop, rend, obs.mask_crop))

    report = {
        "concept": req.concept,
        "material": req.material,
        "fit_gap_mm": gap if "fit_gap" in declared(scad) else None,
        "views": [info | {"iou": best.hybrid.ious[j], "edge_mm": best.hybrid.comparisons[j].chamfer_mm}
                  for j, info in enumerate(infos)],
        "params": final_params,
        "measured": best.draft["measured"],
        "key_params": _provenance(final_params, best.draft),
        "tunable": best.draft["tunable"],
        "questions": best.draft["questions"],
        "through_holes": best.draft.get("through_holes", True),
        "rounds": [{"round": r.index, "done": r.draft["done"], "compile_error": r.compile_error,
                    "spark_iou": r.spark_only.iou if r.spark_only else None,
                    "hybrid_iou": r.hybrid.iou if r.hybrid else None,
                    "edge_mm": r.hybrid.chamfer_mm if r.hybrid else None,
                    "changes": {k: list(v) for k, v in r.hybrid.changes.items()} if r.hybrid else {},
                    "model": r.model, "spark_seconds": r.spark_seconds, "cost_usd": r.cost_usd} for r in rounds],
        "best_round": best.index,
        "iou": best.hybrid.iou,
        "edge_mm": best.hybrid.chamfer_mm,
        "cost_usd": ledger.spent() - spent0,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


async def _segment(photo: Photo, req: PartRequest):
    return await asyncio.gather(
        segment_photo(photo.path, req.sheet_concept, photo.width, photo.height),
        segment_photo(photo.path, req.concept, photo.width, photo.height),
    )


def _provenance(params: dict[str, float], draft: dict) -> list[dict]:
    """Where each number came from, for the result screen."""
    measured, tunable = set(draft["measured"]), set(draft["tunable"])
    rows = []
    for k, v in params.items():
        if k == "fit_gap" or k.startswith("$"):
            continue
        src = "you measured" if k in measured else "fitted from photos" if k in tunable else "from your answers or assumed"
        rows.append({"name": k, "value": v, "source": src})
    order = {"fitted from photos": 0, "you measured": 1, "from your answers or assumed": 2}
    return sorted(rows, key=lambda r: order[r["source"]])
