"""Synthetic phone photos of any SCAD part on an A4 sheet, with known geometry.

Same image model as d1_parallax (12 MP, 26 mm-equivalent lens, textured table, lighting
gradient, lens blur, sensor noise, JPEG), but the part is any mesh and is shaded by
face normal, so edges, holes and walls are visible the way they are in a real photo.
"""

from __future__ import annotations

import cv2
import numpy as np
import trimesh

from repart.camera import A4_MM, Camera, focal_from_35mm, intrinsics
from repart.fit import centre_mesh, place

W_IMG, H_IMG, F35 = 4032, 3024, 26.0
LIGHT = np.array([0.3, -0.4, 0.87]) / np.linalg.norm([0.3, -0.4, 0.87])


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


def camera(target_xy, offset, roll_deg=0.0) -> Camera:
    f = focal_from_35mm(F35, W_IMG, H_IMG)
    t = np.array([*target_xy, 0.0])
    rvec, tvec = look_at(t + np.asarray(offset, float), t, roll_deg)
    return Camera(intrinsics(W_IMG, H_IMG, f), rvec, tvec, W_IMG, H_IMG)


def _background(cam: Camera, rng) -> np.ndarray:
    yy, xx = np.mgrid[0:H_IMG, 0:W_IMG]
    table = 85 + 15 * np.sin(xx / 210.0 + yy / 530.0) + rng.normal(0, 3, (H_IMG, W_IMG))
    img = np.dstack([table * 0.8, table * 0.9, table]).astype(np.float32)
    corners = cam.project(np.array([[0, 0, 0], [A4_MM[0], 0, 0], [A4_MM[0], A4_MM[1], 0], [0, A4_MM[1], 0]]))
    sm = np.zeros((H_IMG * 4, W_IMG * 4), np.uint8)
    cv2.fillPoly(sm, [np.round(corners * 4 * 16).astype(np.int32)], 255, cv2.LINE_8, 4)
    sm = cv2.resize(sm.astype(np.float32) / 255, (W_IMG, H_IMG), interpolation=cv2.INTER_AREA)
    shade = 238 - 10 * (xx / W_IMG)
    return img * (1 - sm[..., None]) + (shade * sm)[..., None]


def _shade_part(img: np.ndarray, mesh: trimesh.Trimesh, cam: Camera, base_bgr, ss: int = 3) -> np.ndarray:
    """Painter's-algorithm render: faces sorted far→near, Lambert shading by normal."""
    v2 = cam.project(mesh.vertices)
    depth = cam.depth(mesh.triangles_center)
    order = np.argsort(-depth)
    lam = np.clip(mesh.face_normals @ LIGHT, 0, 1) * 0.65 + 0.35
    x0, y0 = np.floor(v2.min(0)).astype(int) - 4
    x1, y1 = np.ceil(v2.max(0)).astype(int) + 4
    h, w = (y1 - y0) * ss, (x1 - x0) * ss
    col = np.zeros((h, w, 3), np.float32)
    cov = np.zeros((h, w), np.float32)
    tris = np.round((v2[mesh.faces] - [x0, y0]) * ss * 16).astype(np.int32)
    base = np.asarray(base_bgr, np.float32)
    for i in order:
        c = tuple(float(v) for v in base * lam[i])
        cv2.fillConvexPoly(col, tris[i], c, cv2.LINE_8, 4)
        cv2.fillConvexPoly(cov, tris[i], 1.0, cv2.LINE_8, 4)
    col = cv2.resize(col, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    cov = cv2.resize(cov, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    region = img[y0:y1, x0:x1]
    # col is coverage-premultiplied after the area resize, so composite as premultiplied.
    region[:] = region * (1 - cov[..., None]) + col
    return img


def photo(mesh: trimesh.Trimesh, cam: Camera, xy, yaw_deg: float, rng, base_bgr=(48, 50, 56)) -> np.ndarray:
    """BGR photo of the (centred) mesh placed at xy/yaw on the sheet."""
    img = _background(cam, rng)
    img = _shade_part(img, place(centre_mesh(mesh), xy[0], xy[1], np.radians(yaw_deg)), cam, base_bgr)
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 3, img.shape)
    ok, jpg = cv2.imencode(".jpg", img.clip(0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)
