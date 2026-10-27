"""Day-1 test (needs key + a real photo): SAM 3.1 masks for the sheet and the part.

Checks the three SAM risks from PLAN.md §9:
  - does a text prompt find the sheet and the part (dark/translucent/glare)?
  - mask resolution: is the returned mask downsampled relative to its box?
  - can we recover the camera from the SAM sheet mask (reprojection < 1 px)?

    uv run python spikes/d1_sam_photo.py photo.jpg --part "black plastic knob"
    uv run python spikes/d1_sam_photo.py photo.jpg --part "knob" --sheet "white paper"

Cost: $0.0025 per concept.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import decode_mask_to_raster

from repart import ledger
from repart.camera import CaptureError, focal_from_35mm, sheet_corners, solve_camera
from repart.config import RUNS_DIR
from repart.sam import segment_image


def full_mask(result, width: int, height: int) -> tuple[np.ndarray, list[dict]]:
    """Union of all returned masks in full-frame pixels, plus per-mask resolution stats."""
    out = np.zeros((height, width), np.float32)
    stats = []
    for rec in result.records:
        if rec.kind != "mask":
            continue
        b = rec.bounds
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8).reshape(rec.mask.height, rec.mask.width)
        bw, bh = b.right - b.left, b.bottom - b.top
        x0, y0 = int(round(b.left)), int(round(b.top))
        w, h = max(int(round(bw)), 1), max(int(round(bh)), 1)
        patch = cv2.resize((raster > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        x1, y1 = min(x0 + w, width), min(y0 + h, height)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], patch[: y1 - y0, : x1 - x0])
        stats.append({"object_id": rec.object_id, "confidence": rec.confidence,
                      "box_px": [round(bw, 1), round(bh, 1)], "mask_px": [rec.mask.width, rec.mask.height],
                      "downsample": round(bw / rec.mask.width, 2), "encoding": str(rec.mask.encoding)})
    return out, stats


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--part", required=True, help="short noun phrase for the part")
    ap.add_argument("--sheet", default="white paper sheet")
    ap.add_argument("--focal35", type=float, help="EXIF 35mm-equivalent focal, if known")
    args = ap.parse_args()

    photo = Path(args.photo)
    img = cv2.imread(str(photo))
    H, W = img.shape[:2]
    out = RUNS_DIR / "d1_sam" / photo.stem
    out.mkdir(parents=True, exist_ok=True)

    masks = {}
    for role, concept in (("sheet", args.sheet), ("part", args.part)):
        ledger.ensure(ledger.PRICES["sam-3.1:image"], f"SAM {concept!r}")
        result = await segment_image(photo, concept)
        ledger.record("sam-3.1:image", ledger.PRICES["sam-3.1:image"], concept=concept)
        masks[role], stats = full_mask(result, W, H)
        print(f"{role:5s} {concept!r}: {len(stats)} mask(s)")
        for s in stats:
            print("   ", json.dumps(s))
        np.save(out / f"{role}.npy", masks[role])

    try:
        corners = sheet_corners(masks["sheet"] > 0.5)
        f =focal_from_35mm(args.focal35, W, H) if args.focal35 else None
        cam = solve_camera(corners, W, H, focal_px=f)
        print(f"camera: {cam.center[2]:.0f} mm above sheet, focal {cam.focal:.0f} px, corner reproj {cam.reproj_px:.2f} px")
    except (CaptureError, ValueError) as e:
        print("camera recovery failed:", e)

    vis = img.copy()
    vis[masks["sheet"] > 0.5] = (vis[masks["sheet"] > 0.5] * 0.6 + np.array([255, 200, 0]) * 0.4).astype(np.uint8)
    vis[masks["part"] > 0.5] = (vis[masks["part"] > 0.5] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    cv2.imwrite(str(out / "overlay.jpg"), vis)
    print(f"overlay → {out / 'overlay.jpg'}   spent so far ${ledger.spent():.3f}")


if __name__ == "__main__":
    asyncio.run(main())
