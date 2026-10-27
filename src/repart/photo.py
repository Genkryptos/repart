"""Load a phone photo once, in the orientation the user saw, and keep its lens metadata.

Phones store pixels sideways plus an EXIF rotation flag, and iPhones default to HEIC.
SAM, OpenCV and the camera model must all see the same pixels, so every photo is
normalised into one upright JPEG that everything else reads.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from .camera import focal_from_35mm

register_heif_opener()

EXIF_FOCAL_35MM = 41989  # FocalLengthIn35mmFilm
EXIF_IFD = 0x8769


@dataclass
class Photo:
    path: Path  # normalised upright JPEG
    image: np.ndarray  # BGR
    focal35: float | None  # EXIF 35 mm-equivalent focal length
    source: Path

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def focal_px(self) -> float | None:
        return focal_from_35mm(self.focal35, self.width, self.height) if self.focal35 else None

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]


def load_photo(src: Path, out_dir: Path) -> Photo:
    img = Image.open(src)
    exif = img.getexif()
    f35 = exif.get_ifd(EXIF_IFD).get(EXIF_FOCAL_35MM) or exif.get(EXIF_FOCAL_35MM)
    img = ImageOps.exif_transpose(img).convert("RGB")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "photo.jpg"
    img.save(path, quality=95)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return Photo(path, bgr, float(f35) if f35 else None, src)
