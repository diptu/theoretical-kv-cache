```text
                    ╔══════════════════════════════════════════════════════════════════════╗
                    ║                                                                      ║
                    ║            🧠 THEORETICAL KV CACHE COMPRESSION RESEARCH              ║
                    ║                                                                      ║
                    ║                Mathematical Foundations of Efficient Transformers    ║
                    ║          Memory • Attention • Information • Compression              ║
                    ║                                                                      ║
                    ║     Goal: Understand the true limits of transformer memory usage     ║
                    ║        through linear algebra, probability, and information theory   ║
                    ║                                                                      ║
                    ╚══════════════════════════════════════════════════════════════════════╝
 ```
  
# 🧠 theoretical-kv-cache


## Mathematical Foundations of Efficient Transformers & KV Cache Compression

> A structured research repository exploring the theoretical limits, mathematical structure, and compression mechanisms of KV cache in large language model inference.

---

## 🎯 Research Vision

# Where Should the Bits Go?

**A first-order analysis of KV cache quantization with optimal key–value bit allocation**

[![Paper](https://img.shields.io/badge/paper-in%20preparation-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **TL;DR** — KV cache quantization schemes consistently give keys more bits than values, but the allocation is hand-tuned. We derive a first-order perturbation bound for attention outputs under stochastic uniform quantization that yields a **closed-form optimal key–value bit allocation**, predictable per layer and per head from statistics measurable on a running model. We validate the bound and the allocation rule on Llama 3.2 and Qwen 2.5.

---

## The question

Keys and values are not equally hard to quantize. Empirically, keys need more bits — KIVI quantizes keys per-channel because of outliers, and most production schemes hand keys a higher budget. But *how much* more, and should the gap vary across layers and heads?

The structural reason for the asymmetry is simple once stated:

- **Value error does not amplify.** Attention output is a convex combination `o = a·V` (weights sum to 1), so value perturbations pass through at most linearly.
- **Key error amplifies.** Keys sit inside the softmax: a key perturbation is scaled by the query norm, passed through the softmax map, and then scaled again by the spread of the values.

We turn this observation into a bound, and the bound into an allocation rule.

## Main result (informal)

For attention output `o = softmax(qᵀK/√d)V` under stochastic uniform quantization with `b_K` bits for keys and `b_V` bits for values:

```
‖õ − o‖ ≤ C · (‖q‖/√d) · diam(V) · r_K · 2^(−b_K)  +  r_V · 2^(−b_V)
```

where `r_K`, `r_V` are coordinate ranges of keys and values. Minimizing total bits subject to a target output error gives a closed-form allocation:

```
b_K − b_V ≈ log₂( ‖q‖ · diam(V) / √d ) + log₂( r_K / r_V )
```

Every quantity on the right is cheaply measurable during inference — so the optimal bit gap is a **prediction**, testable per layer and per head, not a tuned hyperparameter.

Formal statements and proofs: [`paper/sections/theory.tex`](paper/) · proof notes: [`paper/notes/`](paper/notes/)

## Results

*(Populated as experiments land — see roadmap below.)*

| Experiment | Models | Status |
|---|---|---|
| Per-layer/head statistics (`‖q‖`, `r_K`, `r_V`, `diam(V)`) | Llama 3.2 1B, Qwen 2.5 1.5B | ⏳ |
| Predicted vs. measured attention output error across bit widths | Llama 3.2 1B | ⏳ |
| Allocation comparison at matched avg. bits: uniform vs. fixed-asymmetric (KIVI-style) vs. **derived** | both | ⏳ |
| Per-layer / per-head allocation heatmaps | both | ⏳ |

Evaluation: perplexity (WikiText-2) + LongBench subsets, at 2/3/4-bit average budgets.

## Repository structure

```
├── paper/            LaTeX source, proof notes, reading notes
├── src/kvbits/       installable package
│   ├── stats.py         per-layer/head statistics hooks
│   ├── quantizers.py    stochastic uniform quantization (per-channel / per-token)
│   ├── allocation.py    closed-form bit allocation rule
│   ├── bounds.py        predicted error from the bound
│   └── patch.py         HF attention patching for on-the-fly KV quantization
├── experiments/      config-driven experiment scripts (one yaml per run)
├── results/          raw run outputs (committed — figures regenerate without a GPU)
├── scripts/          plotting; every paper figure is a script output
└── tests/            quantizer correctness + numerical bound verification
```

## Quickstart

```bash
git clone https://github.com/diptu/theoretical-kv-cache
cd theoretical-kv-cache
pip install -e .

# sanity-check the bound numerically (no GPU needed)
pytest tests/test_bound.py

# collect statistics on a model
python experiments/01_collect_stats.py --config experiments/configs/llama32_1b.yaml
```

## Roadmap

| Tag | Milestone | Target |
|---|---|---|
| `v0.1-lemma` | Perturbation lemma proved + numerical verification in `tests/` | ✅ / ⏳ |
| `v0.2-stats` | Statistics collected on both models; derived allocations + heatmaps | ⏳ |
| `v0.3-results` | Bound-vs-measured plot; three-way allocation comparison | ⏳ |
| `v0.4-draft` | Complete paper draft; all figures via `make figures` | ⏳ |
| `v1.0-arxiv` | arXiv submission | ⏳ |

## Scope

Deliberately narrow: one quantizer family (stochastic uniform), two models, three bit budgets, one theorem, one prediction. Rotation-based schemes (QuaRot, PolarQuant), mixed eviction+quantization, and residual corrections are **out of scope** and discussed as related/future work. Rotations equalize variance *within* vectors; bit allocation exploits asymmetry *between* components (K vs. V, layer vs. layer) — the two compose, which is the natural sequel.

## Related work

Positioning in one line each:

- [**KIVI**](https://arxiv.org/abs/2402.02750) — established the K/V asymmetry empirically; we derive it and predict its magnitude.
- [**KVQuant**](https://arxiv.org/abs/2401.18079) — the empirical statistics landscape (outlier channels, layer sensitivity) our first-order analysis does and doesn't capture.
- [**QJL**](https://arxiv.org/abs/2406.03482) — theorem-first KV quantization via quantized JL sketching; closest in spirit.
- **PolarQuant** (AISTATS 2026) — rotation + residual correction near the information-theoretic limit; orthogonal knob to allocation.
- **KV Cache Transform Coding** (ICLR 2026) — transform-coding view of the cache; our allocation step is reverse water-filling in output-distortion space.

Extended notes: [`paper/notes/reading/`](paper/notes/)

## Open questions beyond this paper

- Outlier-aware bounds: how much of quantization damage does a first-order analysis explain, and what does the residual demand?
- Composition with rotations: allocation *after* variance equalization.
- Information-theoretic lower bounds for allocation under output distortion (vs. per-vector reconstruction).

## Citation

```bibtex
@misc{diptu2026bits,
  title  = {Where Should the Bits Go? A First-Order Analysis of KV Cache
            Quantization with Optimal Key--Value Bit Allocation},
  author = {Nazmul Alam Diptu},
  year   = {2026},
  note   = {In preparation. https://github.com/diptu/theoretical-kv-cache}
}
```

## Contact

Nazmul Alam Diptu · [diptunazmulalam@gmail.com](mailto:diptunazmulalam@gmail.com) · [diptu.github.io](https://diptu.github.io) · [Google Scholar](https://scholar.google.com/citations?user=4e8-JGgAAAAJ)

Working on efficient inference or KV cache quantization? Issues and discussions welcome.