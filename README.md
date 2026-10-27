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

## Spikes

| Script | Needs | What it checks |
|---|---|---|
| `spikes/d1_parallax.py` | nothing | naive vs camera-matched measurement on synthetic photos |
| `spikes/d1_sam_photo.py photo.jpg --part "black knob"` | key + photo | SAM sheet/part masks, mask resolution, camera recovery |
| `spikes/d1_spark_scad.py` | key | Spark one-shot SCAD + 3 feedback rounds, IoU per round |

Every API call goes through `repart.ledger`, a hard spend cap (`REPART_SPEND_CAP_USD`, default $100), because billing alerts don't stop charges.

## Credits

The SAM streaming client is adapted from [meta-models/meta-model-cookbook](https://github.com/meta-models/meta-model-cookbook) (MIT).
