"""Day-1 test (needs key only): can Muse Spark write compilable *parametric* OpenSCAD
from a photo, and does it get closer when shown its own residual overlay?

Uses the synthetic 20°-tilt knob photo from d1_parallax (threshold mask stands in
for SAM), so the only paid calls are Spark's. Runs a one-shot draft plus up to 3
feedback rounds and prints IoU per round: the pass bar is "IoU improves".

    uv run python spikes/d1_spark_scad.py
"""

from __future__ import annotations

import base64
import json
import time

import cv2
import numpy as np

import d1_parallax as d1
from repart import ledger
from repart.camera import Camera, focal_from_35mm, intrinsics, sheet_corners, solve_camera
from repart.compare import compare, residual_overlay
from repart.config import RUNS_DIR, SPARK_MODEL, client
from repart.render import ScadError, full_frame, scad_to_mesh, silhouette

OUT = RUNS_DIR / "d1_spark"
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "scad_draft",
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "what you see and what you changed"},
                "params": {"type": "object", "description": "name -> number, mm",
                           "additionalProperties": {"type": "number"}},
                "scad": {"type": "string", "description": "OpenSCAD; declares every param at top, part sits on z=0, centred on the origin"},
                "questions": {"type": "array", "items": {"type": "string"},
                              "description": "hidden dimensions you need from the user"},
            },
            "required": ["reasoning", "params", "scad", "questions"],
            "additionalProperties": False,
        },
    },
}
SYSTEM = """You are RePart's CAD agent. You reconstruct a replacement for a household plastic part
as *parametric* OpenSCAD. Rules:
- Declare every dimension as a named top-level variable (mm) so an optimiser can tune it.
- The part rests on z=0 and is centred on the origin in x/y. Use $fn >= 96 for round features.
- Prefer simple CSG (cylinder, cube, linear_extrude of polygons, difference for holes).
- Measurements you are given are from a calibrated camera; trust them over your eye.
- When you see a residual overlay: green = matches, red = in the photo but missing from your
  model, blue = in your model but not in the photo. Fix structure, not just numbers."""


def img_url(bgr: np.ndarray, max_side: int = 1400) -> str:
    s = max_side / max(bgr.shape[:2])
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def crop_box(mask: np.ndarray, pad: int = 80):
    x, y, w, h = cv2.boundingRect((mask > 0.5).astype(np.uint8))
    return slice(max(y - pad, 0), y + h + pad), slice(max(x - pad, 0), x + w + pad)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f_true = focal_from_35mm(d1.F35, d1.W_IMG, d1.H_IMG)
    target = np.array([*d1.PART_XY, 0.0])
    rvec, tvec = d1.look_at(target + [60, -95, 310], target, -6)
    img, _ = d1.synth_photo(Camera(intrinsics(d1.W_IMG, d1.H_IMG, f_true), rvec, tvec, d1.W_IMG, d1.H_IMG),
                            np.random.default_rng(0))
    sheet, observed = d1.masks_from_photo(img)
    cam = solve_camera(sheet_corners(sheet), d1.W_IMG, d1.H_IMG)
    mm_px = cam.mm_per_px_at([*d1.PART_XY, 9.0])
    crop = crop_box(observed)

    api = client()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": (
                "Photo of a broken fan-regulator knob on an A4 sheet (cropped). "
                f"Calibrated top-silhouette width ≈ {d1.naive_diameter(cam, observed):.1f} mm "
                "(naive, overestimates tall parts). Height unknown; the camera is tilted ~20°. "
                "Write the first parametric draft.")},
            {"type": "image_url", "image_url": {"url": img_url(img[crop])}},
        ]},
    ]
    for rnd in range(4):
        ledger.ensure(0.05, f"Spark round {rnd}")
        t0 = time.perf_counter()
        resp = api.chat.completions.create(model=SPARK_MODEL, reasoning_effort="high",
                                           response_format=SCHEMA, messages=messages)
        cost = ledger.chat_cost(SPARK_MODEL, resp.usage)
        ledger.record(SPARK_MODEL, cost, spike="d1_spark", round=rnd)
        draft = json.loads(resp.choices[0].message.content)
        (OUT / f"round{rnd}.scad").write_text(draft["scad"])
        print(f"\n== round {rnd}  {time.perf_counter() - t0:.1f}s  ${cost:.4f}  params={draft['params']}")
        print("   ", draft["reasoning"][:300].replace("\n", " "))
        if draft["questions"]:
            print("    asks:", draft["questions"])

        try:
            mesh, _ = scad_to_mesh(draft["scad"])
        except ScadError as e:
            print("    compile error:", str(e)[:200])
            feedback = [{"type": "text", "text": f"OpenSCAD error:\n{e}\nFix the code."}]
        else:
            # Same placement search as the parallax test, but over Spark's mesh, params fixed.
            best = None
            ys, xs = np.nonzero(observed > 0.5)
            for yaw in np.radians(np.arange(0, 360, 15)):
                for dx in (-1.5, 0, 1.5):
                    for dy in (-1.5, 0, 1.5):
                        soft, box = silhouette(d1.place(mesh, d1.PART_XY + [dx, dy], yaw), cam, supersample=2)
                        r = full_frame(soft, box, cam.width, cam.height)
                        c = compare(r, observed, mm_px)
                        if best is None or c.iou > best[0].iou:
                            best = (c, r)
            c, r = best
            print(f"    IoU {c.iou:.4f}  chamfer {c.chamfer_mm:.2f} mm  area ratio {c.area_ratio:.3f}")
            ov = residual_overlay(img, r, observed)[crop]
            cv2.imwrite(str(OUT / f"round{rnd}_overlay.jpg"), ov)
            feedback = [
                {"type": "text", "text": (
                    f"Rendered through the recovered camera: IoU {c.iou:.4f}, mean edge error "
                    f"{c.chamfer_mm:.2f} mm, area ratio {c.area_ratio:.3f} (>1 = too big). Overlay attached. "
                    "Return the improved full draft.")},
                {"type": "image_url", "image_url": {"url": img_url(ov)}},
            ]
        messages += [{"role": "assistant", "content": resp.choices[0].message.content},
                     {"role": "user", "content": feedback}]
    print(f"\nspent so far ${ledger.spent():.3f}")


if __name__ == "__main__":
    main()
