"""RePart web app: phone capture → live fitting view → printable part.

    uv run repart serve            # http://localhost:8000

One job runs at a time (the spend cap and a single OpenSCAD process stay meaningful);
the page polls /api/jobs/<id> for progress.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from . import ledger
from .camera import CaptureError
from .config import ROOT, RUNS_DIR
from .pipeline import FIT_GAPS, PartRequest, make_part

STATIC = Path(__file__).parent / "static"
SAMPLES = ROOT / "data" / "parts"
JOBS_DIR = RUNS_DIR / "jobs"
MAX_PHOTO_BYTES = 30 * 1024 * 1024
PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
MAX_TEXT = 600


@dataclass
class Job:
    id: str
    dir: Path
    request: PartRequest
    state: str = "queued"  # queued | running | retake | done | error
    message: str = ""
    log: list[str] = field(default_factory=list)
    views: list[dict] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)
    report: dict | None = None
    started: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {"id": self.id, "state": self.state, "message": self.message, "log": self.log[-40:],
                "views": self.views, "rounds": self.rounds, "report": self.report,
                "elapsed_s": round(time.time() - self.started, 1), "spent_usd": round(ledger.spent(), 3),
                "cap_usd": ledger.CAP_USD}


app = FastAPI(title="RePart")
jobs: dict[str, Job] = {}
run_lock = threading.Lock()


def _run(job: Job) -> None:
    with run_lock:
        job.state = "running"

        def on_view(j: int, info: dict) -> None:
            job.views.append(info | {"overlay": None})

        def on_round(r) -> None:
            entry = {"round": r.index, "done": r.draft["done"], "reasoning": r.draft["reasoning"],
                     "compile_error": r.compile_error, "questions": r.draft["questions"],
                     "cost_usd": r.cost_usd}
            if r.hybrid:
                entry |= {"spark_iou": r.spark_only.iou, "iou": r.hybrid.iou, "ious": r.hybrid.ious,
                          "edge_mm": r.hybrid.chamfer_mm,
                          "changes": {k: list(v) for k, v in r.hybrid.changes.items()},
                          "overlays": [f"rounds/round{r.index}_view{j + 1}_overlay.jpg" for j in range(len(r.hybrid.ious))]}
            job.rounds.append(entry)

        try:
            job.report = make_part(job.request, log=job.log.append, on_view=on_view, on_round=on_round)
            job.state = "done"
        except CaptureError as e:
            job.state, job.message = "retake", str(e)
        except ledger.SpendCapError as e:
            job.state, job.message = "error", f"Stopped by the spend cap: {e}"
        except Exception as e:  # surface, don't crash the server
            job.state, job.message = "error", f"{type(e).__name__}: {e}"
            job.log.append(traceback.format_exc()[-1500:])


def _clean(text: str, name: str) -> str:
    text = (text or "").strip()
    if len(text) > MAX_TEXT:
        raise HTTPException(400, f"{name} is too long (max {MAX_TEXT} characters).")
    return text


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.post("/api/jobs")
async def create_job(
    photos: list[UploadFile] | None = File(None),
    part: str = Form(...),
    describe: str = Form(""),
    answers: str = Form(""),
    material: str = Form("PETG"),
    reuse: str = Form(""),
) -> dict:
    part = _clean(part, "Part name")
    if not part or len(part.split()) > 6:
        raise HTTPException(400, "Name the part in 1-6 words, like 'black plastic knob'.")
    if material not in FIT_GAPS:
        raise HTTPException(400, f"Material must be one of {', '.join(FIT_GAPS)}.")
    jid = secrets.token_hex(6)
    jdir = JOBS_DIR / jid
    inputs = jdir / "inputs"
    inputs.mkdir(parents=True)
    paths = []
    if reuse:  # re-run with answers: same photos, SAM results come from cache
        src = jobs.get(reuse)
        if not src:
            raise HTTPException(404, "Original job not found.")
        paths = list(src.request.photos)
    else:
        photos = photos or []
        if not 1 <= len(photos) <= 3:
            raise HTTPException(400, "Upload 1 to 3 photos.")
        for k, up in enumerate(photos):
            ext = Path(up.filename or "").suffix.lower()
            if ext not in PHOTO_TYPES:
                raise HTTPException(400, f"Photo {k + 1}: use JPG, PNG or HEIC.")
            data = await up.read()
            if len(data) > MAX_PHOTO_BYTES:
                raise HTTPException(400, f"Photo {k + 1} is over 30 MB.")
            p = inputs / f"photo{k + 1}{ext}"
            p.write_bytes(data)
            paths.append(p)
    ans = [a.strip() for a in _clean(answers, "Answers").splitlines() if a.strip()]
    req = PartRequest(photos=paths, concept=part, out_dir=jdir / "out", description=_clean(describe, "Description"),
                      answers=ans, material=material)
    job = Job(jid, jdir, req)
    jobs[jid] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str) -> dict:
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job.public()


@app.get("/api/jobs/{jid}/files/{path:path}")
def job_file(jid: str, path: str) -> FileResponse:
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Job not found.")
    root = (job.dir / "out").resolve()
    target = (root / path).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(404, "File not found.")
    return FileResponse(target)


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> dict:
    """Spoken description → text, via Muse Voice Transcribe."""
    from .voice import VoiceError, transcribe

    data = await audio.read()
    if not data or len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "Recording is empty or over 10 MB.")
    tmp = JOBS_DIR / "voice" / f"{secrets.token_hex(6)}{Path(audio.filename or 'a.webm').suffix or '.webm'}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    try:
        return transcribe(tmp)
    except (VoiceError, ledger.SpendCapError) as e:
        raise HTTPException(400, str(e)) from e
    finally:
        tmp.unlink(missing_ok=True)


@app.get("/api/samples")
def samples() -> list[dict]:
    """Bundled example parts, so anyone can try the app without a broken part to hand."""
    out = []
    for spec in sorted(SAMPLES.glob("*/part.json")):
        d = json.loads(spec.read_text())
        if d.get("sample"):
            out.append({"id": spec.parent.name, "title": d["sample"], "photos": d.get("photos") or [d["photo"]],
                        "concept": d["concept"], "description": d.get("description", ""),
                        "answers": d.get("answers", []), "material": d.get("material", "PETG")})
    return out


@app.get("/api/samples/{sid}/{name}")
def sample_file(sid: str, name: str) -> FileResponse:
    root = SAMPLES.resolve()
    target = (root / sid / name).resolve()
    if root not in target.parents or target.suffix.lower() not in PHOTO_TYPES or not target.is_file():
        raise HTTPException(404, "Sample not found.")
    return FileResponse(target)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
