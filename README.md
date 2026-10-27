# RePart

Film a broken household plastic part on an A4 sheet. RePart recovers the phone camera from the sheet, segments the part with **SAM 3.1**, and has a **Muse Spark** agent write parametric OpenSCAD. It renders the model *through the same camera*, compares the result with the photo, and keeps fixing it until the dimensions match. The output is a printable STL.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). All code in this repository was written during the event.

## Why "through the same camera"

A part's top face is closer to the lens than the sheet, so it looks bigger. Flattening the sheet and measuring overestimates a 32 mm, 18 mm-tall knob by **2–5 mm**. Comparing silhouettes in the camera's own image recovers it within **0.015 mm** on synthetic photos (`spikes/d1_parallax.py`).

## Setup

```bash
uv sync
cp .env.example .env    # add MODEL_API_KEY
brew install --cask openscad@snapshot
```

## Make a part

Put the part on a white A4 sheet on a darker matte surface. Take two photos with all four sheet corners in frame: one tilted ~20° from about 30 cm, and one steep (~45–50°) from another side. The steep one pins down heights. Moving the part between shots is fine.

```bash
uv run repart make top.jpg steep.jpg --part "black plastic knob" \
  --describe "Fan regulator knob; the D-shaped hole fits the shaft." \
  --answer "Shaft hole 6 mm with a flat 4.5 mm across, 10 mm deep." \
  --material PETG
```

This writes `part.stl`, `part.scad` (every dimension named), `fit_overlay.jpg` and `report.json` to `runs/parts/<photo>/`. When the photo can't show something, RePart lists the questions to answer with `--answer`.

## Web app

```bash
uv run repart serve                 # http://localhost:8000
uv run repart serve --host 0.0.0.0  # reach it from a phone on the same Wi-Fi
```

Capture (two photo slots, description, optional measurements, Speak buttons via Muse Voice Transcribe) → live fitting view (overlays per round, overlap and edge error, Spark's reasoning) → result (3D preview, every dimension labelled by where it came from, STL/SCAD download, follow-up questions that re-run the part). "Try a sample part" loads bundled photos, so you can try it without a broken part. The microphone needs https on a phone; localhost is fine.

### Hosting

```bash
docker build -t repart .
docker run -p 8000:8000 -e MODEL_API_KEY=... -e REPART_ACCESS_CODE=pick-one repart
```

A public URL spends real API credit, so paid endpoints are guarded: an optional access code (share `https://host/?code=...`), per-IP hourly limits (`REPART_JOBS_PER_HOUR`, default 6), a 24-hour budget (`REPART_DAILY_CAP_USD`, default $10) under the overall cap, a short queue, and day-old jobs deleted.

## Tests

```bash
uv run pytest -q    # offline, ~1 s: camera recovery (incl. steep shots), SCAD helpers, fallbacks, lint, eval, web guards
```

## Measure accuracy

Each eval part is a folder in `data/parts/<id>/` with the photo and a `part.json` holding caliper readings (format in `src/repart/evaluate.py`).

```bash
uv run repart eval          # → runs/eval/summary.md
```

## Spikes

| Script | Needs | What it checks |
|---|---|---|
| `spikes/d1_parallax.py` | nothing | naive vs camera-matched measurement on synthetic photos |
| `spikes/d1_sam_photo.py photo.jpg --part "black knob"` | key + photo | SAM sheet/part masks, mask resolution, camera recovery |
| `spikes/d1_spark_scad.py` | key | Spark one-shot SCAD + 3 feedback rounds, IoU per round |

Every API call goes through `repart.ledger`, a hard spend cap (`REPART_SPEND_CAP_USD`, default $100), because billing alerts don't stop charges.

## License

MIT, see [LICENSE](LICENSE).

## Credits

The SAM streaming client is adapted from [meta-models/meta-model-cookbook](https://github.com/meta-models/meta-model-cookbook) (MIT).
