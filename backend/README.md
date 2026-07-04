# Phase 0 backend: model & trace API

Implements `Plan.md`'s Phase 0 — a tiny instrumented GPT plus a FastAPI service
that exposes real, per-layer forward-pass activations (including KV cache
growth across autoregressive decode steps) for the frontend visualization to
render. See `../plan.md` for the overall architecture this feeds into.

## Toy task

A 3-layer / 3-head / 48-dim GPT trained to sort a length-6 sequence of digits
drawn from `{0, 1, 2}` (displayed as letters `A`/`B`/`C`) — the same toy task
used by Karpathy's minGPT and bbycroft's llm-viz nano demo model.

## Setup

Dependencies (`fastapi`, `uvicorn`, `torch`, `pydantic`, `numpy`) — install
with `pip install -r requirements.txt` if not already present.

## Commands

Run all commands from this `backend/` directory.

```bash
# Train the model (takes well under a minute on CPU) -> checkpoints/model.pt + config.json
python3 -m app.train

# Precompute traces for the fixed EXAMPLES set -> traces/example_*.json
python3 -m app.export_traces

# Run the API (requires a trained checkpoint)
python3 -m uvicorn app.main:app --reload --port 8000
```

## API

- `GET /health` — liveness check.
- `GET /config` — model architecture (`vocab_size`, `block_size`, `n_layer`,
  `n_head`, `n_embd`, `sort_length`, `vocab_symbols`).
- `GET /examples` — the fixed example token sequences.
- `POST /trace` — body `{"tokens": [2,0,1,2,1,0], "steps": 5}` (`steps`
  optional, clamped to what fits in the context window: `block_size -
  len(tokens)`). Returns a versioned trace: `{version, config, prefill,
  decode_steps}`.

### Trace format (`version: "1.0"`)

- `prefill`: full forward pass over the input tokens — `token_emb`, `pos_emb`,
  `embed_sum`, per-layer (`ln1_out`, `q`, `k_new`, `v_new`, `attn_scores`,
  `attn_pattern`, `attn_out`, `resid_after_attn`, `mlp_hidden`, `mlp_out`,
  `resid_after_mlp`), `final_ln_out`, `logits`.
- `decode_steps`: one entry per generated token — `new_token`,
  `cache_size_after`, per-layer `layers_new_kv` (`k_new`/`v_new`/
  `attn_pattern` **for that step only**, not the whole cache), and `logits`.
  The frontend reconstructs full KV cache state at any step by accumulating
  `k_new`/`v_new` across `prefill` + prior `decode_steps` — this keeps the
  payload small instead of repeating the whole cache every step.

Note: because `block_size = 2*sort_length - 1 = 11`, decoding is capped at
`block_size - len(prompt)` steps — for a full 6-token prompt that's 5 decode
steps, which together with the token implied by the prefill's own final
logits recovers all 6 sorted digits. This is an inherent boundary of the toy
model's fixed context window, not a bug.
