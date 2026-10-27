"""SAM 3.1 over Meta Model API: segment a concept in an image or track it through a video.

Request/stream handling adapted from meta-models/meta-model-cookbook
(07_sam/01_api_basics/python/sam_api.py), MIT License, Copyright (c) Meta Platforms, Inc.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
from collections import defaultdict
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from meta_sam_parser import (
    ResponsesEventLike,
    SegmentationResult,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)

from .config import CACHE_DIR, SAM_MODEL, async_client
from .tracking import Track, load_tracks, mask_stats, save_tracks


def _media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed is None:
        raise ValueError(f"Could not infer a media type for {path}.")
    return guessed


def _input(concept: str, media: dict[str, str]) -> list[dict[str, Any]]:
    phrase = concept.strip()
    if not phrase:
        raise ValueError("The concept must be a short noun phrase.")
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": phrase}, media],
        }
    ]


async def _events(stream: AsyncIterable[Any]) -> AsyncIterator[ResponsesEventLike]:
    """Plain event dicts for the parser. The API sends `response.content_part.done`
    but not `response.output_text.done`, which the parser needs to close a lane,
    so synthesize it."""
    done: set[tuple[object, object, object]] = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in done:
                done.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in done
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            done.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            }


def _validate(result: SegmentationResult) -> None:
    if result.outcome.status != "completed":
        reason = getattr(result.outcome, "reason", "unknown")
        detail = getattr(result.outcome, "detail", None)
        raise RuntimeError(f"Segmentation incomplete: {reason} ({detail})")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise RuntimeError("; ".join(f"line {d.line}: {d.message}" for d in errors))


async def segment_image(path: Path, concept: str) -> SegmentationResult:
    url = f"data:{_media_type(path)};base64,{base64.b64encode(path.read_bytes()).decode()}"
    async with async_client() as api:
        stream = await api.responses.create(
            model=SAM_MODEL,
            stream=True,
            input=_input(concept, {"type": "input_image", "image_url": url}),
        )
        async with stream, aclosing(_events(stream)) as events:
            result = await parse_responses_stream(
                events, image_segmentation_format()
            ).final_result()
    _validate(result)
    return result


async def upload_video(path: Path) -> str:
    async with async_client() as api:
        uploaded = await api.files.create(
            file=(path.name, path.read_bytes(), _media_type(path)), purpose="user_data"
        )
    if not str(uploaded.id).startswith("file-"):
        raise ValueError("The Files API returned an invalid file ID.")
    return uploaded.id


async def segment_video(file_id: str, concept: str) -> SegmentationResult:
    async with async_client() as api:
        stream = await api.responses.create(
            model=SAM_MODEL,
            stream=True,
            input=_input(concept, {"type": "input_video", "file_id": file_id}),
        )
        async with stream, aclosing(_events(stream)) as events:
            parsed = parse_responses_stream(events, video_segmentation_format())
            async with parsed:
                async for _ in parsed:
                    pass
                result = await parsed.final_result()
    _validate(result)
    return result


def tracks_from_result(result: SegmentationResult, concept: str) -> list[Track]:
    """One Track per SAM object id, using mask centroids (box centre if no mask)."""
    rows: dict[str, dict[int, tuple]] = defaultdict(dict)
    for rec in result.records:
        frame = -1 if rec.frame is None else rec.frame.frame_index
        if rec.kind == "mask":
            b = rec.bounds
            raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8)
            raster = raster.reshape(rec.mask.height, rec.mask.width) > 0
            stats = mask_stats(raster, b.left, b.top, b.right, b.bottom)
            if stats:
                rows[rec.object_id][frame] = (*stats, b.left, b.top, b.right, b.bottom)
        elif rec.kind == "box" and frame not in rows[rec.object_id]:
            cx, cy = (rec.left + rec.right) / 2, (rec.top + rec.bottom) / 2
            area = (rec.right - rec.left) * (rec.bottom - rec.top)
            rows[rec.object_id][frame] = (cx, cy, area, rec.left, rec.top, rec.right, rec.bottom)

    tracks = []
    for oid, by_frame in sorted(rows.items(), key=lambda kv: int(kv[0])):
        frames = sorted(by_frame)
        a = np.array([by_frame[f] for f in frames], dtype=float)
        tracks.append(
            Track(oid, concept, np.array(frames), a[:, 0], a[:, 1], a[:, 2], a[:, 3:7])
        )
    return tracks


@dataclass
class ImageMask:
    object_id: str
    mask: np.ndarray  # full-frame float 0..1
    confidence: float | None
    downsample: float  # box px per mask px; > 1 means SAM returned a coarser mask

    @property
    def area(self) -> float:
        return float(self.mask.sum())

    def centroid(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.mask > 0.5)
        return float(xs.mean()), float(ys.mean())


def image_masks(result: SegmentationResult, width: int, height: int) -> list[ImageMask]:
    """Each returned object as a full-frame soft mask (box-local raster scaled to its bounds)."""
    out = []
    for rec in result.records:
        if rec.kind != "mask":
            continue
        b = rec.bounds
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8).reshape(rec.mask.height, rec.mask.width)
        x0, y0 = int(np.floor(b.left)), int(np.floor(b.top))
        x1, y1 = min(int(np.ceil(b.right)), width), min(int(np.ceil(b.bottom)), height)
        if x1 <= x0 or y1 <= y0:
            continue
        # Resample in float so a coarse mask gets soft (sub-pixel) edges, not blocky ones.
        patch = cv2.resize((raster > 0).astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        full = np.zeros((height, width), np.float32)
        full[y0:y1, x0:x1] = patch
        out.append(ImageMask(rec.object_id, full, rec.confidence, (b.right - b.left) / rec.mask.width))
    return out


async def segment_photo(path: Path, concept: str, width: int, height: int) -> list[ImageMask]:
    """SAM on a still, cached on disk by (image hash, concept), behind the spend cap."""
    from . import ledger

    cache = CACHE_DIR / "sam_image" / f"{_sha(path)}-{concept.replace(' ', '_')}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return [ImageMask(str(i), m, c, d) for i, m, c, d in zip(z["ids"], z["masks"], z["conf"], z["down"])]
    price = ledger.PRICES["sam-3.1:image"]
    ledger.ensure(price, f"SAM {concept!r}")
    masks = image_masks(await segment_image(path, concept), width, height)
    ledger.record("sam-3.1:image", price, concept=concept)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        ids=np.array([m.object_id for m in masks]),
        masks=np.array([m.mask.astype(np.float16) for m in masks]).reshape(len(masks), height, width),
        conf=np.array([m.confidence if m.confidence is not None else np.nan for m in masks]),
        down=np.array([m.downsample for m in masks]),
    )
    return masks


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


async def track_concept(video: Path, concept: str, refresh: bool = False) -> list[Track]:
    """Upload + segment + convert, cached on disk by (video hash, concept)."""
    key = f"{_sha(video)}-{concept.replace(' ', '_')}"
    cache = CACHE_DIR / f"{key}.tracks.json"
    if cache.exists() and not refresh:
        return load_tracks(cache)
    file_id_cache = CACHE_DIR / f"{_sha(video)}.file_id"
    if file_id_cache.exists():
        file_id = file_id_cache.read_text().strip()
    else:
        file_id = await upload_video(video)
        file_id_cache.parent.mkdir(parents=True, exist_ok=True)
        file_id_cache.write_text(file_id)
    tracks = tracks_from_result(await segment_video(file_id, concept), concept)
    save_tracks(cache, tracks)
    return tracks
