# Requirements

Environment, dependencies, and compute needed to reproduce every run in `preregistration.md`.

## Environment

- Python ≥ 3.11
- CUDA-capable GPU, ≥ 8 GB VRAM (T4 sufficient; all runs sized for free/low-cost tiers)
- Linux (tested) · ~10 GB disk for checkpoints + eigenspectra

## Dependencies

```
torch>=2.4          # pin exact version in requirements.lock at freeze
torchvision
numpy
scipy
pyhessian           # Hessian eigenspectrum: Lanczos + Hutchinson
matplotlib          # analysis/ figures
pandas              # results aggregation
pyyaml              # configs/
```

```bash
pip install -r requirements.txt
pip freeze > requirements.lock   # committed at pre-registration freeze
```

**Version policy:** `requirements.txt` states intent; `requirements.lock` (frozen with the pre-registration commit) is authoritative for confirmatory runs. Confirmatory results must be produced from the locked environment.

**DST implementation:** RigL, implemented in `src/` against the locked torch version (no external DST framework dependency — keeps the mask logic auditable). Correctness check: reproduce RigL paper's ResNet-CIFAR numbers within ±0.5% before Phase 1.

## Compute budget

| Phase | Runs | GPU-hrs (T4 est.) | Fits on |
|---|---|---|---|
| Pilot (§8) | 4 | ~5 | Colab free / Kaggle |
| Phase 1 | 60 | ~60–70 | Colab Pro / Kaggle (30 hr/wk) / GSU cluster |
| Phases 2–3 | ~300 | ~300 | GSU cluster or spot instances |

Per-run cost = training (~45 min) + Hessian measurement (~10–15 min: top-50 Lanczos + 100-probe Hutchinson). Estimates validated during pilot; corrected numbers recorded here before freeze.

## Reproducibility

- Every run driven by a config file in `configs/` — no CLI-only hyperparameters
- Seeds fixed per cell (3 seeds: 0, 1, 2); condition (d) inherits its parent (c) seed
- Per-run outputs to `results/{run_id}/`: metrics.json, top-k eigenvalues, trace + CI, spectral density, final mask hash
- Figure scripts in `analysis/` regenerate all plots from `results/` alone

## Data

CIFAR-10 via torchvision (auto-download). Symmetric label noise generated deterministically from the run seed; noisy label files committed per noise level so all methods train on *identical* corrupted labels.
