"""Write the synthetic 20°-tilt knob photo as an eval part (data/parts/synthetic_knob).

No EXIF, so the pipeline has to estimate the focal length from the sheet. Scored
against the true geometry: 32 mm diameter, 20 mm total height (18 body + 2 ridge).

    uv run python spikes/make_synthetic_part.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import d1_parallax as d1
from repart.camera import Camera, focal_from_35mm, intrinsics

OUT = Path(__file__).resolve().parents[1] / "data" / "parts" / "synthetic_knob"


def main() -> None:
    f = focal_from_35mm(d1.F35, d1.W_IMG, d1.H_IMG)
    t = np.array([*d1.PART_XY, 0.0])
    rvec, tvec = d1.look_at(t + [60, -95, 310], t, -6)
    img, _ = d1.synth_photo(Camera(intrinsics(d1.W_IMG, d1.H_IMG, f), rvec, tvec, d1.W_IMG, d1.H_IMG),
                            np.random.default_rng(0))
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "top_tilt.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (OUT / "part.json").write_text(json.dumps({
        "photo": "top_tilt.jpg",
        "concept": "black round knob",
        "description": "Synthetic render of a fan-regulator knob on an A4 sheet (flat shading, no texture).",
        "answers": ["There is a raised pointer ridge on top of the knob: 2 mm tall, 3 mm wide, "
                    "running from near the centre to about 3 mm from the edge."],
        "material": "PETG",
        "calipers": [
            {"name": "diameter (mid-height)", "kind": "diameter_at", "z": 9, "mm": 32.0},
            {"name": "diameter (near base)", "kind": "diameter_at", "z": 3, "mm": 32.0},
            {"name": "total height", "kind": "height", "mm": 20.0},
        ],
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
