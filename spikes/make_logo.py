"""RePart logo: a knob seen from above with a chunk of its rim broken off, the green
replacement piece sliding back into place, and the D-shaped shaft hole.

One geometry, two outputs:
  src/repart/static/logo.svg                       rounded square, for the web page and favicon
  src/repart/static/apple-touch-icon.png           180 px, for "Add to Home Screen"
  ios/RePartCapture/Assets.xcassets/AppIcon...     1024 px opaque PNG (iOS rounds the corners)

    uv run python spikes/make_logo.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
S = 1024
C = S / 2
R, R0 = 340, 200            # knob radius; where the broken chunk starts
A0, A1 = -75.0, -5.0        # chunk angles (degrees, clockwise from +x, y down)
GAP_DEG, GAP_PX = 3.0, 14   # crack around the chunk
SHIFT = 34                  # the new piece sits a little out, sliding in
HOLE_R, HOLE_FLAT = 92, 58  # D-shaped shaft hole: circle with a flat on the right
BG = ("#1c2b23", "#0a0f0c")
BODY, GREEN = "#e8ede9", "#30d158"
MID = math.radians((A0 + A1) / 2)
DX, DY = SHIFT * math.cos(MID), SHIFT * math.sin(MID)


def _pt(r: float, a: float, dx: float = 0, dy: float = 0) -> str:
    t = math.radians(a)
    return f"{C + dx + r * math.cos(t):.1f} {C + dy + r * math.sin(t):.1f}"


def _sector(r0: float, r1: float, a0: float, a1: float, dx: float = 0, dy: float = 0) -> str:
    return (f"M{_pt(r0, a0, dx, dy)} L{_pt(r1, a0, dx, dy)} A{r1} {r1} 0 0 1 {_pt(r1, a1, dx, dy)} "
            f"L{_pt(r0, a1, dx, dy)} A{r0} {r0} 0 0 0 {_pt(r0, a0, dx, dy)} Z")


def svg(rounded: bool = True) -> str:
    h = math.sqrt(HOLE_R**2 - HOLE_FLAT**2)
    hole = f"M{C + HOLE_FLAT:.1f} {C - h:.1f} A{HOLE_R} {HOLE_R} 0 1 0 {C + HOLE_FLAT:.1f} {C + h:.1f} Z"
    crack = _sector(R0 - GAP_PX, R + 10, A0 - GAP_DEG, A1 + GAP_DEG)
    rx = 224 if rounded else 0
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" role="img" aria-label="RePart">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{BG[0]}"/><stop offset="1" stop-color="{BG[1]}"/></linearGradient>
    <mask id="body"><circle cx="{C}" cy="{C}" r="{R}" fill="#fff"/><path d="{crack}" fill="#000"/><path d="{hole}" fill="#000"/></mask>
  </defs>
  <rect width="{S}" height="{S}" rx="{rx}" fill="url(#bg)"/>
  <rect width="{S}" height="{S}" fill="{BODY}" mask="url(#body)"/>
  <path d="{_sector(R0, R, A0, A1, DX, DY)}" fill="{GREEN}"/>
</svg>
"""


def png(size: int = S, ss: int = 4) -> Image.Image:
    """Same shapes, rasterised with 4× supersampling (no SVG renderer needed)."""
    n = size * ss
    k = S / n
    y, x = (np.mgrid[0:n, 0:n] + 0.5) * k
    t = (x + y) / (2 * S)
    bg = np.array([int(BG[0][i:i + 2], 16) for i in (1, 3, 5)]) * (1 - t[..., None]) + \
        np.array([int(BG[1][i:i + 2], 16) for i in (1, 3, 5)]) * t[..., None]

    def polar(dx: float = 0, dy: float = 0):
        u, v = x - C - dx, y - C - dy
        return np.hypot(u, v), np.degrees(np.arctan2(v, u)), u

    r, a, u = polar()
    hole = (r <= HOLE_R) & (u <= HOLE_FLAT)
    crack = (a >= A0 - GAP_DEG) & (a <= A1 + GAP_DEG) & (r >= R0 - GAP_PX)
    body = (r <= R) & ~hole & ~crack
    r2, a2, _ = polar(DX, DY)
    green = (r2 >= R0) & (r2 <= R) & (a2 >= A0) & (a2 <= A1)

    out = bg.copy()
    out[body] = [int(BODY[i:i + 2], 16) for i in (1, 3, 5)]
    out[green] = [int(GREEN[i:i + 2], 16) for i in (1, 3, 5)]
    out = out.reshape(size, ss, size, ss, 3).mean(axis=(1, 3))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def main() -> None:
    (ROOT / "src/repart/static/logo.svg").write_text(svg())
    png(180).save(ROOT / "src/repart/static/apple-touch-icon.png")
    assets = ROOT / "ios/RePartCapture/Assets.xcassets"
    icon = assets / "AppIcon.appiconset"
    icon.mkdir(parents=True, exist_ok=True)
    png().save(icon / "AppIcon-1024.png")
    (assets / "Contents.json").write_text(json.dumps({"info": {"author": "xcode", "version": 1}}, indent=2))
    (icon / "Contents.json").write_text(json.dumps({
        "images": [{"filename": "AppIcon-1024.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"}],
        "info": {"author": "xcode", "version": 1},
    }, indent=2))
    logo = assets / "Logo.imageset"  # shown in the app's header
    logo.mkdir(exist_ok=True)
    png(256).save(logo / "Logo.png")
    (logo / "Contents.json").write_text(json.dumps({
        "images": [{"filename": "Logo.png", "idiom": "universal"}], "info": {"author": "xcode", "version": 1},
    }, indent=2))
    print("wrote logo.svg, apple-touch-icon.png, AppIcon-1024.png and Logo.png")


if __name__ == "__main__":
    main()
