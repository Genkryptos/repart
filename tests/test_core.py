"""Fast offline tests (no API calls): camera, SCAD helpers, fallbacks, lint, eval, web guards."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))

import synth  # noqa: E402
from repart import lint  # noqa: E402
from repart.camera import A4_MM, CaptureError, sheet_corners, solve_camera  # noqa: E402
from repart.evaluate import measure  # noqa: E402
from repart.pipeline import concept_fallbacks  # noqa: E402
from repart.render import ScadError, scad_to_mesh  # noqa: E402
from repart.scad import bake_params, declared  # noqa: E402

SHEET = np.array([[0, 0, 0], [A4_MM[0], 0, 0], [A4_MM[0], A4_MM[1], 0], [0, A4_MM[1], 0]], float)


def _sheet_mask(cam) -> np.ndarray:
    m = np.zeros((cam.height, cam.width), np.uint8)
    cv2.fillPoly(m, [np.round(cam.project(SHEET)).astype(np.int32)], 1)
    return m


@pytest.mark.parametrize("offset,roll", [([60, -95, 310], -6), ([-200, 160, 230], 8), ([30, -45, 290], 4)])
def test_camera_recovered_from_sheet(offset, roll):
    """Includes the steep 48° shot where the 297 mm side looks shorter than the 210 mm side."""
    true = synth.camera((150, 100), offset, roll)
    for focal in (true.focal, None):  # EXIF focal known, and estimated from the sheet
        cam = solve_camera(sheet_corners(_sheet_mask(true)), true.width, true.height, focal_px=focal)
        assert cam.reproj_px < 1.0
        assert abs(cam.center[2] - true.center[2]) < 2.0
        assert abs(cam.focal - true.focal) / true.focal < 0.01


def test_cut_off_sheet_is_rejected():
    true = synth.camera((150, 100), [10, -25, 200], 2)
    with pytest.raises(CaptureError, match="cut off"):
        sheet_corners(_sheet_mask(true))


def test_bake_params_rewrites_and_prepends():
    code = "D = 30;\n  H=10 ;\ncylinder(d=D, h=H);"
    out = bake_params(code, {"D": 32.5, "H": 18, "fit_gap": 0.2})
    assert "D = 32.5;" in out and "H = 18;" in out and out.startswith("fit_gap = 0.2;")
    assert declared(out) >= {"D", "H", "fit_gap"}


def test_scad_errors_are_readable():
    with pytest.raises(ScadError):
        scad_to_mesh("cylinder(d=10 h=5);")


def test_position_params_may_go_negative():
    from repart.fit import is_position

    assert all(is_position(n) for n in ["hole_cx", "hole_offset", "pin_x", "y_shift", "pos_y", "x0"])
    assert not any(is_position(n) for n in ["wall_H", "base_L", "hole_d", "depth", "hex", "max_h", "box"])


def test_concept_fallbacks():
    assert concept_fallbacks("black round knob") == ["black round knob", "black knob", "knob", "black object"]
    assert concept_fallbacks("knob") == ["knob"]


def test_lint_flags_overhang_but_not_a_clean_part():
    clean = scad_to_mesh("$fn=96; cylinder(d=32,h=18); translate([1,-1.5,18]) cube([12,3,2]);")[0]
    bad = scad_to_mesh("$fn=96; cylinder(d=32,h=18); translate([6.5,-1.5,18]) cube([13,3,2]);")[0]
    assert lint.notes(clean) == []
    dist, clock = lint.overhang(bad)
    assert dist == pytest.approx(3.6, abs=0.2) and clock == 3


def test_measurements():
    washer = scad_to_mesh("$fn=128; difference(){ cylinder(d=24,h=6); translate([0,0,-1]) cylinder(d=8,h=8); }")[0]
    assert measure(washer, "diameter_at", 3) == pytest.approx(24, abs=0.05)
    assert measure(washer, "hole_diameter_at", 3) == pytest.approx(8, abs=0.05)
    assert measure(washer, "height") == pytest.approx(6, abs=1e-6)


@pytest.mark.parametrize("scad,holes", [
    ("$fn=96; difference(){ cylinder(d=24,h=6); translate([0,0,-1]) cylinder(d=8,h=8); }", 1),
    ("$fn=96; cylinder(d=24,h=6);", 0),
])
def test_carve_holes_restores_through_holes(scad, holes):
    """SAM returns solid masks; holes showing the sheet must be carved back, and none invented."""
    from repart.holes import carve_holes

    cam = synth.camera((150, 100), [60, -95, 310], -6)
    img = synth.photo(scad_to_mesh(scad)[0], cam, (150, 100), 35, np.random.default_rng(0))
    dark = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 140).astype(np.uint8) * _sheet_mask(cam)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(dark)
    cv2.fillPoly(solid, [max(cnts, key=cv2.contourArea)], 1)  # SAM-style: no holes
    _, rep = carve_holes(solid.astype(np.float32), img, _sheet_mask(cam).astype(np.float32))
    assert rep.holes == holes


def test_web_guards(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from repart import web

    client = TestClient(web.app)
    monkeypatch.setattr(web, "ACCESS_CODE", "s3cret")
    assert client.post("/api/jobs", data={"part": "knob"}).status_code == 403
    monkeypatch.setattr(web, "JOBS_PER_HOUR", 1)
    web._hits.clear()
    h = {"X-Access-Code": "s3cret"}
    # First request passes the guards and fails validation (no photos); the second is rate limited.
    assert client.post("/api/jobs", data={"part": "knob"}, headers=h).status_code == 400
    assert client.post("/api/jobs", data={"part": "knob"}, headers=h).status_code == 429
