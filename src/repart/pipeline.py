"""Photo in, printable part out: the one path the CLI and the web app both call."""

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
from .photo import load_photo
from .render import scad_to_mesh
from .sam import ImageMask, segment_photo
from .scad import bake_params, declared

# Radial clearance (mm) added to mating holes/sockets per material. FDM rules of thumb;
# calibrate on the first test print and override with fit_gap.
FIT_GAPS = {"PLA": 0.15, "PETG": 0.20, "ABS": 0.20, "Nylon": 0.25, "TPU": 0.10}


@dataclass
class PartRequest:
    photo: Path
    concept: str  # SAM noun phrase for the part, e.g. "black plastic knob"
    out_dir: Path
    description: str = ""
    answers: list[str] = field(default_factory=list)
    sheet_concept: str = "white paper sheet"
    material: str = "PETG"
    fit_gap: float | None = None
    max_rounds: int = 4
    focal35: float | None = None  # override when the photo has no EXIF


def _inside(poly: np.ndarray, pt: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(poly.astype(np.float32), pt, False) >= 0


def _pick_part(masks: list[ImageMask], corners: np.ndarray, concept: str) -> ImageMask:
    on_sheet = [m for m in masks if m.area > 0 and _inside(corners, m.centroid())]
    if not on_sheet:
        raise CaptureError(f"Couldn't find a '{concept}' on the sheet. Try a different description, "
                           "or a background that contrasts with the part.")
    return max(on_sheet, key=lambda m: m.area)


def make_part(req: PartRequest, log=print) -> dict:
    out = req.out_dir
    out.mkdir(parents=True, exist_ok=True)
    spent0 = ledger.spent()
    photo = load_photo(req.photo, out)
    log(f"photo {photo.width}×{photo.height}, EXIF focal {photo.focal35 or 'none'} mm-equiv")

    sheets, parts = asyncio.run(_segment(photo, req))
    if not sheets:
        raise CaptureError("Couldn't find the A4 sheet. Use plain white paper on a darker surface.")
    sheet = max(sheets, key=lambda m: m.area)
    corners = sheet_corners(sheet.mask > 0.5)
    f35 = req.focal35 or photo.focal35
    focal_px = focal_from_35mm(f35, photo.width, photo.height) if f35 else None
    try:
        cam = solve_camera(corners, photo.width, photo.height, focal_px=focal_px)
    except ValueError as e:
        raise CaptureError("Can't work out the camera from this photo. Tilt the phone 15-20° "
                           "instead of shooting straight down.") from e
    part = _pick_part(parts, corners, req.concept)
    obs = Observation(photo.image, part.mask, cam)
    log(f"sheet found (corner fit {cam.reproj_px:.2f} px), camera {cam.center[2]:.0f} mm up, "
        f"tilt {obs.tilt_deg():.0f}°; part mask {part.area:.0f} px, SAM downsample {part.downsample:.2f}×")

    rounds = agent.run(obs, req.description or f"Photo of a broken household part ({req.concept}) on an A4 sheet.",
                       answers=req.answers, max_rounds=req.max_rounds, out_dir=out / "rounds", log=log)
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
    cv2.imwrite(str(out / "fit_overlay.jpg"), residual_overlay(obs.image_crop, best.hybrid.rendered, obs.mask_crop))

    report = {
        "concept": req.concept,
        "material": req.material,
        "fit_gap_mm": gap if "fit_gap" in declared(scad) else None,
        "camera": {"height_mm": float(cam.center[2]), "tilt_deg": obs.tilt_deg(), "focal_px": cam.focal,
                   "focal_source": "exif" if focal_px else "estimated", "corner_reproj_px": cam.reproj_px,
                   "focal35_exif": f35},
        "sam": {"part_downsample": part.downsample, "part_confidence": part.confidence,
                "sheet_downsample": sheet.downsample, "n_part_masks": len(parts)},
        "params": final_params,
        "measured": best.draft["measured"],
        "tunable": best.draft["tunable"],
        "questions": best.draft["questions"],
        "rounds": [{"round": r.index, "done": r.draft["done"], "compile_error": r.compile_error,
                    "spark_iou": r.spark_only.iou if r.spark_only else None,
                    "hybrid_iou": r.hybrid.iou if r.hybrid else None,
                    "edge_mm": r.hybrid.comparison.chamfer_mm if r.hybrid else None,
                    "changes": {k: list(v) for k, v in r.hybrid.changes.items()} if r.hybrid else {},
                    "spark_seconds": r.spark_seconds, "cost_usd": r.cost_usd} for r in rounds],
        "best_round": best.index,
        "iou": best.hybrid.iou,
        "edge_mm": best.hybrid.comparison.chamfer_mm,
        "cost_usd": ledger.spent() - spent0,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


async def _segment(photo, req: PartRequest):
    return await asyncio.gather(
        segment_photo(photo.path, req.sheet_concept, photo.width, photo.height),
        segment_photo(photo.path, req.concept, photo.width, photo.height),
    )
