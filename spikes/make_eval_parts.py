"""Render shaded two-photo eval parts with known geometry into data/parts/syn_*.

Each part gets a ~20° photo and a steep ~48° photo from the other side (moved and
rotated in between) plus a part.json whose calipers are the true SCAD dimensions.
Answers are left empty on purpose: these test what the photos alone recover.

    uv run python spikes/make_eval_parts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import synth
from repart.render import scad_to_mesh

PARTS = Path(__file__).resolve().parents[1] / "data" / "parts"
XY = np.array([150.0, 100.0])

SPECS = {
    "syn_washer": {
        "scad": "$fn=128; difference(){ cylinder(d=24, h=6); translate([0,0,-1]) cylinder(d=8, h=8); }",
        "concept": "black plastic washer",
        "description": "Plastic spacer washer from a ceiling fan mount.",
        "calipers": [
            {"name": "outer diameter", "kind": "diameter_at", "z": 3, "mm": 24.0},
            {"name": "hole diameter", "kind": "hole_diameter_at", "z": 3, "mm": 8.0},
            {"name": "thickness", "kind": "height", "mm": 6.0},
        ],
    },
    "syn_l_bracket": {
        "scad": """$fn=96;
difference(){
  union(){ translate([-20,-10,0]) cube([40,20,3]); translate([17,-10,0]) cube([3,20,20]); }
  for (x=[-12,4]) translate([x,0,-1]) cylinder(d=5, h=5);
}""",
        "concept": "black plastic bracket",
        "description": "Small L-shaped mounting bracket with two screw holes in the base.",
        "calipers": [
            {"name": "base length", "kind": "length", "mm": 40.0},
            {"name": "width", "kind": "width", "mm": 20.0},
            {"name": "height", "kind": "height", "mm": 20.0},
            {"name": "screw hole", "kind": "hole_diameter_at", "z": 1.5, "mm": 5.0},
        ],
    },
    "syn_c_clip": {
        "scad": """$fn=128;
difference(){ cylinder(d=20, h=8); translate([0,0,-1]) cylinder(d=14, h=10); translate([-3,4,-1]) cube([6,10,10]); }""",
        "concept": "black plastic clip",
        "description": "C-shaped clip from a cooler; it snaps onto a pipe.",
        "calipers": [
            {"name": "outer diameter", "kind": "length", "mm": 20.0},
            {"name": "height", "kind": "height", "mm": 8.0},
        ],
    },
}

VIEWS = [  # (camera offset from part mm, roll, part offset on sheet mm, part yaw deg)
    ([60, -95, 310], -6, (0.0, 0.0), 35),
    ([-200, 160, 230], 8, (-12.0, 9.0), 80),
]


def main() -> None:
    for pid, spec in SPECS.items():
        mesh, _ = scad_to_mesh(spec["scad"])
        out = PARTS / pid
        out.mkdir(parents=True, exist_ok=True)
        names = []
        for i, (off, roll, dxy, yaw) in enumerate(VIEWS):
            xy = XY + dxy
            img = synth.photo(mesh, synth.camera(xy, off, roll), xy, yaw, np.random.default_rng(10 + i))
            name = f"photo{i + 1}.jpg"
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            names.append(name)
        (out / "part.json").write_text(json.dumps({
            "photos": names, "concept": spec["concept"], "description": spec["description"],
            "answers": [], "material": "PETG", "calipers": spec["calipers"],
            "truth_scad": spec["scad"],
        }, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
