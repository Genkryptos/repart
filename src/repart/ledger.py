"""Hard spend cap. Meta's billing alerts don't stop charges, so every API call goes
through here: refuse before the call if the estimate would cross the cap, record
the actual cost after."""

from __future__ import annotations

import json
import os
import time

from .config import RUNS_DIR

CAP_USD = float(os.environ.get("REPART_SPEND_CAP_USD", "100"))
LEDGER = RUNS_DIR / "spend.jsonl"

# USD, pay-as-you-go Standard mode (dev.meta.ai/docs/pricing-rate-limits, Oct 2026).
PRICES = {
    "muse-spark-1.3": {"in": 1.25e-6, "cached": 0.15e-6, "out": 4.25e-6},
    "muse-wonder-1.0": {"in": 4.00e-6, "cached": 0.20e-6, "out": 20.0e-6},
    "sam-3.1:image": 2.50e-3,
    "sam-3.1:frame": 0.20e-3,  # video billed for at least 50 frames per request
    "muse-image-1.0": 0.01,
    "muse-voice-transcribe-1.0:second": 0.18 / 3600,
}


class SpendCapError(RuntimeError):
    pass


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line)


def spent_since(t0: float) -> float:
    if not LEDGER.exists():
        return 0.0
    rows = (json.loads(line) for line in LEDGER.read_text().splitlines() if line)
    return sum(r["usd"] for r in rows if r["t"] >= t0)


def ensure(estimate_usd: float, what: str) -> None:
    total = spent()
    if total + estimate_usd > CAP_USD:
        raise SpendCapError(f"{what}: ${total:.2f} spent + ${estimate_usd:.3f} would exceed the ${CAP_USD:.0f} cap.")


def record(what: str, usd: float, **meta) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"t": time.time(), "what": what, "usd": usd, **meta}) + "\n")


def chat_cost(model: str, usage) -> float:
    p = PRICES[model]
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return (usage.prompt_tokens - cached) * p["in"] + cached * p["cached"] + usage.completion_tokens * p["out"]
