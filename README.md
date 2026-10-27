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

Put the part on a white A4 sheet on a darker matte surface. Tilt the phone 15–20°, about 30 cm away, with all four sheet corners in frame.

```bash
uv run repart make photo.jpg --part "black plastic knob" \
  --describe "Fan regulator knob; the D-shaped hole fits the shaft." \
  --answer "Shaft hole 6 mm with a flat 4.5 mm across, 10 mm deep." \
  --material PETG
```

This writes `part.stl`, `part.scad` (every dimension named), `fit_overlay.jpg` and `report.json` to `runs/parts/<photo>/`. When the photo can't show something, RePart lists the questions to answer with `--answer`.

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
