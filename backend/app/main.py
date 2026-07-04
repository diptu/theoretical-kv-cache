import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import GPTConfig, SORT_LENGTH, VOCAB_SYMBOLS
from .model import GPT
from .tracing import trace_payload

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

app = FastAPI(title="theoretical-kv-cache trace API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3002"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fixed example prompts (digit ids into NUM_DIGITS), used both by the
# frontend's example picker and by export_traces.py for static traces.
EXAMPLES = [
    [2, 0, 1, 2, 1, 0],
    [0, 1, 2, 0, 1, 2],
    [2, 2, 1, 0, 0, 1],
]

_model: GPT | None = None


def get_model() -> GPT:
    global _model
    if _model is None:
        config_path = CHECKPOINT_DIR / "config.json"
        weights_path = CHECKPOINT_DIR / "model.pt"
        if not config_path.exists() or not weights_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model checkpoint not found. Run `python -m app.train` first.",
            )
        with open(config_path) as f:
            cfg_dict = json.load(f)
        config = GPTConfig(**cfg_dict)
        model = GPT(config)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        _model = model
    return _model


def config_dict(model: GPT) -> dict:
    return {
        "vocab_size": model.config.vocab_size,
        "block_size": model.config.block_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "sort_length": SORT_LENGTH,
        "vocab_symbols": VOCAB_SYMBOLS,
    }


class TraceRequest(BaseModel):
    tokens: list[int]
    steps: int | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def get_config():
    return config_dict(get_model())


@app.get("/examples")
def get_examples():
    return {"examples": EXAMPLES}


@app.post("/trace")
def post_trace(req: TraceRequest):
    model = get_model()
    if len(req.tokens) == 0 or any(t < 0 or t >= model.config.vocab_size for t in req.tokens):
        raise HTTPException(status_code=400, detail="tokens must be within [0, vocab_size)")
    if len(req.tokens) > model.config.block_size:
        raise HTTPException(status_code=400, detail="too many tokens for model context window")

    max_steps = model.config.block_size - len(req.tokens)
    steps = max_steps if req.steps is None else max(0, min(req.steps, max_steps))

    return trace_payload(model, config_dict(model), req.tokens, steps)
