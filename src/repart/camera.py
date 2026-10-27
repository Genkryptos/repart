"""Recover the phone camera (intrinsics + pose) from an A4 sheet in the photo.

World frame: the sheet lies in z = 0 with corners (0,0), (297,0), (297,210), (0,210) mm,
and +z points up toward the camera, so a part of height h occupies 0 <= z <= h.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

A4_MM = (297.0, 210.0)
ID1_CARD_MM = (85.60, 53.98)  # ISO/IEC 7810 ID-1: bank cards, PVC Aadhaar; for close-ups


class CaptureError(ValueError):
    """The photo can't be measured; the message tells the user how to retake it."""
SENSOR_DIAG_35MM = float(np.hypot(36.0, 24.0))


@dataclass
class Camera:
    K: np.ndarray  # 3x3
    rvec: np.ndarray  # 3x1
    tvec: np.ndarray  # 3x1
    width: int
    height: int
    reproj_px: float = 0.0

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def center(self) -> np.ndarray:
        return (-self.R.T @ self.tvec).ravel()

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    def project(self, pts: np.ndarray) -> np.ndarray:
        img, _ = cv2.projectPoints(np.asarray(pts, dtype=np.float64), self.rvec, self.tvec, self.K, None)
        return img.reshape(-1, 2)

    def depth(self, pts: np.ndarray) -> np.ndarray:
        cam = (self.R @ np.asarray(pts, dtype=np.float64).T + self.tvec).T
        return cam[:, 2]

    def mm_per_px_at(self, pt: np.ndarray) -> float:
        """Approximate size of one pixel, in mm, at a world point's depth."""
        return float(self.depth(np.atleast_2d(pt))[0] / self.focal)

    def plane_homography(self) -> np.ndarray:
        """Image px -> sheet-plane mm (z = 0)."""
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.tvec.ravel()])
        return np.linalg.inv(H / H[2, 2])


def intrinsics(width: int, height: int, focal_px: float) -> np.ndarray:
    return np.array([[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1]], dtype=np.float64)


def focal_from_35mm(f35: float, width: int, height: int) -> float:
    """EXIF FocalLengthIn35mmFilm -> focal length in pixels (diagonal convention)."""
    return f35 * float(np.hypot(width, height)) / SENSOR_DIAG_35MM


def focal_from_homography(H: np.ndarray, width: int, height: int) -> float | None:
    """Single-view focal estimate (square pixels, principal point at centre).

    H maps sheet-plane coordinates to image px. With the principal point moved to the
    origin, the first two columns of K^-1 H are orthogonal; solving that for f gives
    f^2 = -(h1x h2x + h1y h2y) / (h1z h2z). Ill-conditioned for near-frontal views.
    """
    T = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]], dtype=np.float64)
    h = T @ H
    h1, h2 = h[:, 0], h[:, 1]
    denom = h1[2] * h2[2]
    if abs(denom) < 1e-12:
        return None
    f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / denom
    return float(np.sqrt(f2)) if f2 > 0 else None


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise in the image, starting so side 0->1 is a long side."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    q = pts[np.argsort(ang)]
    if np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[2] - q[1]):
        q = np.roll(q, -1, axis=0)
    return q


def sheet_corners(mask: np.ndarray) -> np.ndarray:
    """Sub-pixel sheet corners from a sheet mask.

    Rough quad from the largest contour, then a robust line fit per side (so a part
    covering an edge or corner doesn't break it), intersected pairwise.
    """
    m = (mask > 0).astype(np.uint8)
    edge = 3
    if m[:edge].any() or m[-edge:].any() or m[:, :edge].any() or m[:, -edge:].any():
        raise CaptureError("The sheet is cut off at the photo edge. Move the phone back until all four corners are visible.")
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No sheet found in mask.")
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        raise ValueError("Sheet outline is not a quadrilateral.")
    quad = _order_quad(quad)

    pts = cnt.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        n = np.array([-d[1], d[0]]) / L
        s = (pts - a) @ d / L**2
        near = (np.abs((pts - a) @ n) < 0.01 * peri) & (s > 0.1) & (s < 0.9)
        side = pts[near] if near.sum() >= 10 else np.array([a, b])
        vx, vy, x0, y0 = cv2.fitLine(side.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append(np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0]))
    corners = []
    for i in range(4):
        p = np.cross(lines[i - 1], lines[i])
        corners.append(p[:2] / p[2])
    return _order_quad(np.array(corners))


def solve_camera(
    corners_px: np.ndarray,
    width: int,
    height: int,
    focal_px: float | None = None,
    sheet_mm: tuple[float, float] = A4_MM,
) -> Camera:
    """Pose (and focal, if not given) from 4 ordered sheet corners.

    Which image side is the sheet's long side is ambiguous: in a steep shot the 297 mm
    side can look shorter than the 210 mm one. Both assignments are solved and the one
    that reprojects the corners best wins; the wrong aspect leaves hundreds of px of
    error even when the focal is estimated, since that estimate uses only orthogonality.
    """
    W, Hh = sheet_mm
    obj = np.array([[0, 0, 0], [W, 0, 0], [W, Hh, 0], [0, Hh, 0]], dtype=np.float64)
    cands = []
    for shift in (0, 1):
        img = np.roll(corners_px, -shift, axis=0)
        f = focal_px
        if f is None:
            Hm = cv2.getPerspectiveTransform(obj[:, :2].astype(np.float32), img.astype(np.float32))
            f = focal_from_homography(Hm, width, height)
            if f is None:
                continue
        cam = _pose(obj, img, intrinsics(width, height, f), width, height)
        if cam is not None:
            cands.append(cam)
    if not cands:
        if focal_px is None:
            raise ValueError("Cannot estimate focal length from a near-frontal view; pass EXIF focal.")
        raise ValueError("Could not recover a camera above the sheet.")
    return min(cands, key=lambda c: c.reproj_px)


def _pose(obj: np.ndarray, corners_px: np.ndarray, K: np.ndarray, width: int, height: int) -> Camera | None:
    # Corner order fixes the world's handedness. If the camera lands below the sheet,
    # use the reflected order (still the same side first) so +z points at the camera.
    for order in ([0, 1, 2, 3], [1, 0, 3, 2]):
        img = corners_px[order]
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        # IPPE alone leaves ~10 px residuals on 12 MP photos; LM refinement gets < 1 px.
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
        cam = Camera(K, rvec, tvec, width, height)
        if cam.center[2] > 0:
            cam.reproj_px = float(np.sqrt(np.mean(np.sum((cam.project(obj) - img) ** 2, axis=1))))
            return cam
    return None
