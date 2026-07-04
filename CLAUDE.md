# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This is primarily a **research/documentation project** (see "What this project is" below). The only
implemented code so far is `backend/` (Phase 0 of `plan.md`'s visualization platform — see "Visualization
backend" below); everything else described by `Requirements.md` (the Next.js/MDX/KaTeX platform) and most
of `plan.md`'s phases are not yet built. Do not invent frontend tooling or assume it's wired up until it
actually appears in the repo; check `ls` / `git status` before assuming files or directories described
below exist.

## Commands

Run from `backend/` (see `backend/README.md` for full details):

```bash
python3 -m app.train              # train the toy GPT -> checkpoints/model.pt + config.json
python3 -m app.export_traces       # precompute traces for the fixed EXAMPLES -> traces/example_*.json
python3 -m uvicorn app.main:app --reload --port 8000   # run the trace API (needs a trained checkpoint)
```

No lint/test suite is configured yet for `backend/`.

## What this project is

`theoretical-kv-cache` (see `README.md`) is a long-term academic preparation project studying the
mathematical foundations of efficient transformer inference, centered on KV cache compression:

- Linear algebra, probability/statistics, and information theory as applied to transformer memory usage.
- Attention mechanisms and efficient variants (MQA, GQA, FlashAttention, PagedAttention, StreamingLLM, H2O,
  SnapKV, KIVI).
- KV cache redundancy, low-rank/sparse approximation, token pruning/eviction, and information-theoretic
  compression limits.

The long-term goal is a theoretical framework for KV cache compression with provable guarantees. The
current phase is foundational study (no original research results yet) — the emphasis is on structured
notes, paper analysis, and manual derivations rather than code.

The README describes an intended repository layout (`mathematics/`, `attention/`, `kv-cache/`,
`paper_notes/`, `derivations/`, `research/`) organized by topic. These directories do not exist yet as of
this writing — treat that layout as the target structure to create/follow when adding new research notes,
not as something to search for.

## Planned platform (not yet built)

`Requirements.md` specifies a future "Interactive Research Platform" — a Next.js site that turns the
Markdown research notes into a "Live Paper": MDX rendering of notes, KaTeX for math, an in-browser
reactive simulation sandbox (sliders/inputs driving Math.js/TensorFlow.js computations synced via
Zustand), D3.js visualizations, Framer Motion animation, Fuse.js search, and a print-to-PDF academic
export mode. Deployment is intended to be Vercel + GitHub integration with ISR.

Treat this as a **spec for future work, not an existing implementation** — when asked to build toward it,
check first whether any of this scaffolding has since been added (e.g. `package.json`, an `app/` or
`pages/` directory) rather than assuming the Phase 1 stack from `Requirements.md` is already in place.

`plan.md` narrows and adapts that platform to this repo's actual research focus: an interactive
visualization of GPT-style inference and KV cache behavior specifically (in the spirit of
[bbycroft.net/llm](https://bbycroft.net/llm)), rather than a general-purpose model explorer. Its intended
frontend stack is Next.js + React Three Fiber + drei + Zustand (not yet started); the backend it depends on
is already implemented (see below).

## Visualization backend (`backend/`, implemented)

A FastAPI service (`backend/app/`) implementing `plan.md`'s Phase 0. It trains a tiny instrumented GPT
(3 layers, 3 heads, 48-dim, in `backend/app/model.py`) on a toy digit-sorting task (`backend/app/dataset.py`)
and exposes real per-layer forward-pass activations — embeddings, Q/K/V, attention scores/patterns, MLP
activations, logits — plus **KV cache growth across autoregressive decode steps** via `POST /trace`
(`backend/app/tracing.py`, `backend/app/main.py`). `backend/app/export_traces.py` precomputes traces for a
fixed example set into `backend/traces/*.json` for static frontend consumption. The trace JSON format is
versioned (`version: "1.0"`) and documented in `backend/README.md` — read that file before changing the
payload shape, since the (not-yet-built) frontend's `DataService` (plan.md Phase 3) depends on it.

The trained checkpoint (`backend/checkpoints/*.pt`, `config.json`) is gitignored and regenerated via
`python3 -m app.train`; only the exported example traces are meant to be committed as fixture data.

## Working conventions

- Research notes are Markdown, one concept/paper per file, following the "Research Workflow" in
  `README.md`: study a concept → read papers → write structured paper notes → derive key equations →
  connect math to transformers → log open questions.
- When adding paper notes or derivations, mirror the existing analytical framing used in the README
  (mathematical formulation, computational complexity, memory behavior, strengths/limitations, open
  problems) rather than inventing a new note format.
