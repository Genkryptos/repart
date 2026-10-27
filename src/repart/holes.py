"""Put through-holes back into SAM masks.

SAM returns each object as a solid region: a washer comes back as a disc. A through-hole
shows the sheet behind it, so inside the part mask, enclosed regions that look like the
sheet are carved out (softly, by how sheet-like each pixel is). Skipped when the part
doesn't contrast with the sheet, where the colours can't tell a hole from the part.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_CONTRAST = 40  # grey levels between part and sheet
MIN_HOLE_PX = 25


@dataclass
class HoleReport:
    holes: int
    carved_px: float
    skipped: str = ""
    areas_px: list[float] = field(default_factory=list)


def carve_holes(part: np.ndarray, image_bgr: np.ndarray, sheet: np.ndarray) -> tuple[np.ndarray, HoleReport]:
    p = part > 0.5
    if p.sum() < 100:
        return part, HoleReport(0, 0, "part too small")
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = np.ones((15, 15), np.uint8)
    near_part = cv2.dilate(p.astype(np.uint8), k, iterations=3) > 0
    sheet_px = g[(sheet > 0.5) & ~near_part]
    core = cv2.erode(p.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    part_px = g[core]
    if len(sheet_px) < 1000 or len(part_px) < 50:
        return part, HoleReport(0, 0, "not enough sheet or part pixels")
    s_tone, p_tone = float(np.median(sheet_px)), float(np.median(part_px))
    if abs(s_tone - p_tone) < MIN_CONTRAST:
        return part, HoleReport(0, 0, f"part and sheet too similar ({p_tone:.0f} vs {s_tone:.0f})")

    sheetness = np.clip((g - p_tone) / (s_tone - p_tone), 0, 1)  # 0 = part tone, 1 = sheet tone
    cand = ((sheetness > 0.5) & p).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    # A hole is enclosed: none of it touches the outline of the part.
    rim = p & ~(cv2.erode(p.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_HOLE_PX and not rim[lab == i].any():
            keep[i] = True
    holes = keep[lab]
    if not holes.any():
        return part, HoleReport(0, 0)
    # Soft edge: grow the region a little and weight by how sheet-like each pixel is.
    region = cv2.dilate(holes.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = part.astype(np.float32).copy()
    out[region] *= 1.0 - sheetness[region]
    areas = [float(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n) if keep[i]]
    return out, HoleReport(int(keep.sum()), float((part - out).sum()), areas_px=areas)
