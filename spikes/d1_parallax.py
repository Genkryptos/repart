"""Day-1 test: does camera-matched comparison beat naive sheet rectification?

Renders synthetic phone photos (12 MP, 26 mm-equivalent lens) of a known knob on an
A4 sheet from several heights/tilts, then measures the knob diameter two ways:

  naive    rectify the sheet plane with a homography, measure the part's outline in mm
  matched  recover the camera from the sheet corners, render the parametric SCAD
           through it, and optimise D (+ x, y, yaw) to maximise silhouette IoU

Masks come from intensity thresholds here, standing in for SAM 3.1. Pass bar:
matched within 0.3 mm. No API key needed.

    uv run python spikes/d1_parallax.py
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
from scipy.optimize import minimize

from repart.camera import A4_MM, Camera, CaptureError, focal_from_35mm, intrinsics, sheet_corners, solve_camera
from repart.compare import compare, residual_overlay, soft_iou
from repart.config import RUNS_DIR
from repart.render import full_frame, scad_to_mesh, silhouette

KNOB = """
D = 32; H = 18; ridge_w = 3; ridge_h = 2;
$fn = 160;
cylinder(d = D, h = H);
translate([-ridge_w / 2, 0, H]) cube([ridge_w, D / 2 - 3, ridge_h]);
"""
TRUE = {"D": 32.0, "H": 18.0}
PART_XY = np.array([150.0, 100.0])  # true placement on the sheet, mm
PART_YAW = np.radians(35)
W_IMG, H_IMG = 4032, 3024
F35 = 26.0
OUT = RUNS_DIR / "d1"


def look_at(center: np.ndarray, target: np.ndarray, roll_deg: float = 0) -> tuple[np.ndarray, np.ndarray]:
    z = (target - center) / np.linalg.norm(target - center)
    hint = np.array([0.0, 1.0, 0.0])
    y = hint - (hint @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.vstack([x, y, z])
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    return cv2.Rodrigues(R)[0], (-R @ center).reshape(3, 1)


def place(mesh, xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    m = mesh.copy()
    m.vertices = m.vertices @ Rz.T + [xy[0], xy[1], 0]
    return m


def synth_photo(cam: Camera, rng) -> tuple[np.ndarray, np.ndarray]:
    """BGR photo + true part coverage mask."""
    yy, xx = np.mgrid[0:H_IMG, 0:W_IMG]
    table = 85 + 15 * np.sin(xx / 210.0 + yy / 530.0) + rng.normal(0, 3, (H_IMG, W_IMG))
    img = np.dstack([table * 0.8, table * 0.9, table]).astype(np.float32)
    sheet = cam.project(np.array([[0, 0, 0], [A4_MM[0], 0, 0], [A4_MM[0], A4_MM[1], 0], [0, A4_MM[1], 0]]))
    sm = np.zeros((H_IMG * 4, W_IMG * 4), np.uint8)
    cv2.fillPoly(sm, [np.round(sheet * 4 * 16).astype(np.int32)], 255, cv2.LINE_8, 4)
    sm = cv2.resize(sm.astype(np.float32) / 255, (W_IMG, H_IMG), interpolation=cv2.INTER_AREA)
    shade = 238 - 10 * (xx / W_IMG)  # mild lighting gradient
    img = img * (1 - sm[..., None]) + (shade * sm)[..., None]
    mesh, _ = scad_to_mesh(KNOB, TRUE)
    soft, box = silhouette(place(mesh, PART_XY, PART_YAW), cam, supersample=6)
    part = full_frame(soft, box, W_IMG, H_IMG)
    img = img * (1 - part[..., None]) + np.array([45, 48, 52]) * part[..., None]
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 3, img.shape)
    ok, jpg = cv2.imencode(".jpg", img.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR), part


def masks_from_photo(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Threshold stand-ins for SAM: sheet (bright) and a soft part mask (dark, inside the sheet)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sheet = (g > 170).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sheet)
    sheet = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    hull = np.zeros_like(sheet)
    cnts, _ = cv2.findContours(sheet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(hull, [cv2.convexHull(max(cnts, key=cv2.contourArea))], 1)
    hull = cv2.erode(hull, np.ones((9, 9), np.uint8))
    part = np.clip((225 - g) / (225 - 55), 0, 1) * hull  # soft coverage between sheet and part tones
    part[part < 0.05] = 0
    return sheet, part


def naive_diameter(cam: Camera, part: np.ndarray) -> float:
    """Rectify the part mask onto the sheet plane and take the area-equivalent diameter,
    after removing the ridge by a morphological opening in mm space (0.1 mm/px)."""
    Himg2mm = cam.plane_homography()
    S = np.diag([10.0, 10.0, 1.0])  # 0.1 mm per rectified pixel
    rect = cv2.warpPerspective(part, S @ Himg2mm, (int(A4_MM[0] * 10), int(A4_MM[1] * 10)))
    core = cv2.morphologyEx((rect > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    return float(2 * np.sqrt(core.sum() / np.pi) / 10.0)


def fit_matched(cam: Camera, observed: np.ndarray, d0: float, h_known: float | None):
    """Optimise D (and H if unknown) + placement to maximise IoU."""
    # Initial placement: observed centroid back-projected to the mid-height plane.
    ys, xs = np.nonzero(observed > 0.5)
    ray = np.linalg.inv(cam.K) @ np.array([xs.mean(), ys.mean(), 1.0])
    ray_w = cam.R.T @ ray
    C = cam.center
    zmid = (h_known or 15.0) / 2
    lam = (zmid - C[2]) / ray_w[2]
    xy0 = (C + lam * ray_w)[:2]
    evals = {"n": 0, "compile": 0.0}

    def render(D, H, x, y, yaw):
        mesh, dt = scad_to_mesh(KNOB, {"D": round(D, 3), "H": round(H, 3)})
        evals["compile"] += dt
        soft, box = silhouette(place(mesh, (x, y), yaw), cam)
        return full_frame(soft, box, cam.width, cam.height)

    def loss(p):
        evals["n"] += 1
        D, x, y, yaw = p[:4]
        H = h_known if h_known is not None else p[4]
        return 1 - soft_iou(render(D, H, x, y, yaw), observed)

    h0 = [] if h_known is not None else [15.0]
    yaws = np.radians(np.arange(0, 360, 30))
    best_yaw = min(yaws, key=lambda a: loss([d0, *xy0, a, *h0]))
    p0 = np.array([d0, *xy0, best_yaw, *h0])
    step = np.array([1.0, 1.0, 1.0, np.radians(5), *([2.0] if h0 else [])])
    res = minimize(loss, p0, method="Nelder-Mead",
                   options={"initial_simplex": np.vstack([p0, p0 + np.diag(step)]),
                            "xatol": 1e-3, "fatol": 1e-6, "maxfev": 600})
    D, x, y, yaw = res.x[:4]
    H = h_known if h_known is not None else res.x[4]
    rendered = render(D, H, x, y, yaw)
    return {"D": float(D), "H": float(H), "xy": [float(x), float(y)], "yaw_deg": float(np.degrees(yaw) % 360),
            "iou": 1 - float(res.fun), "evals": evals["n"], "compile_s": evals["compile"]}, rendered


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    f_true = focal_from_35mm(F35, W_IMG, H_IMG)
    target = np.array([PART_XY[0], PART_XY[1], 0.0])
    scenarios = [  # (name, camera offset from part in mm, roll)
        ("top-down 300mm", [0, 0, 300], 0),
        ("tilt ~10deg 300mm", [30, -45, 290], 4),
        ("tilt ~20deg 330mm", [60, -95, 310], -6),
        ("too close 200mm", [10, -25, 200], 2),  # sheet cut off: must be rejected
    ]
    rows = []
    for name, off, roll in scenarios:
        center = target + np.array(off, dtype=float)
        rvec, tvec = look_at(center, target, roll)
        true_cam = Camera(intrinsics(W_IMG, H_IMG, f_true), rvec, tvec, W_IMG, H_IMG)
        img, true_part = synth_photo(true_cam, rng)
        sheet, part = masks_from_photo(img)

        try:
            corners = sheet_corners(sheet)
        except CaptureError as e:
            print(f"\n== {name}: rejected → {e}")
            rows.append({"scenario": name, "rejected": str(e)})
            continue
        cam_exif = solve_camera(corners, W_IMG, H_IMG, focal_px=f_true)
        try:
            f_est = solve_camera(corners, W_IMG, H_IMG).focal
        except ValueError:
            f_est = float("nan")
        mm_px = cam_exif.mm_per_px_at([*PART_XY, TRUE["H"]])

        d_naive = naive_diameter(cam_exif, part)
        t0 = time.perf_counter()
        fit_h, rend = fit_matched(cam_exif, part, d_naive, h_known=TRUE["H"])
        t_fit = time.perf_counter() - t0
        fit_free, _ = fit_matched(cam_exif, part, d_naive, h_known=None)
        cmp = compare(rend, part, mm_px)

        row = {
            "scenario": name,
            "cam_height_mm": float(cam_exif.center[2]),
            "corner_reproj_px": cam_exif.reproj_px,
            "focal_est_err_pct": 100 * (f_est - f_true) / f_true,
            "mm_per_px": mm_px,
            "D_naive": d_naive,
            "D_naive_err_mm": d_naive - TRUE["D"],
            "D_matched_Hknown": fit_h["D"],
            "D_matched_Hknown_err_mm": fit_h["D"] - TRUE["D"],
            "D_matched_Hfree": fit_free["D"],
            "H_matched_Hfree": fit_free["H"],
            "iou": fit_h["iou"],
            "chamfer_mm": cmp.chamfer_mm,
            "fit_seconds": t_fit,
            "evals": fit_h["evals"],
        }
        rows.append(row)
        print(f"\n== {name}: camera {row['cam_height_mm']:.0f} mm up, {mm_px * 1000:.0f} µm/px, "
              f"corner reproj {row['corner_reproj_px']:.2f} px, focal-from-sheet error {row['focal_est_err_pct']:+.1f}%")
        print(f"   naive rectified D = {d_naive:.2f} mm  (error {row['D_naive_err_mm']:+.2f} mm)")
        print(f"   matched, H known  D = {fit_h['D']:.3f} mm  (error {row['D_matched_Hknown_err_mm']:+.3f} mm)  "
              f"IoU {fit_h['iou']:.4f}  chamfer {cmp.chamfer_mm * 1000:.0f} µm  "
              f"{fit_h['evals']} evals, {t_fit:.1f}s")
        print(f"   matched, H free   D = {fit_free['D']:.3f} mm, H = {fit_free['H']:.2f} mm (true 18)")

        x0, y0, bw, bh = cv2.boundingRect((part > 0.5).astype(np.uint8))
        pad = 60
        crop = (slice(max(y0 - pad, 0), y0 + bh + pad), slice(max(x0 - pad, 0), x0 + bw + pad))
        cv2.imwrite(str(OUT / f"{name.split()[0]}_{int(row['cam_height_mm'])}_overlay.png"),
                    np.hstack([img[crop], residual_overlay(img, rend, part)[crop]]))
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
