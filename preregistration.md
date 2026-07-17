# Pre-Registration: Effective Dimension Predicts the Sparse Double Descent Peak

**Author:** Nazmul Alam · **Date frozen:** [YYYY-MM-DD, set at commit] · **Status:** DRAFT — freeze after pipeline validation (§8)

This document fixes hypotheses, conditions, measurements, and analysis *before* any confirmatory run. After freezing, it is edited only via dated entries in §9 (Deviations).

---

## 1. Hypotheses

**H1 (primary).** The location of the sparse double descent (SDD) test-error peak is governed by effective dimension (d_eff), not nominal sparsity. Networks trained with different sparsification methods peak at different nominal sparsities but at the same d_eff.

**H2 (mechanism isolation).** A frozen final DST mask, retrained from scratch, lands on the same collapsed curve as its parent DST run — i.e., the mask's effect on the peak is fully mediated by d_eff, not by DST training dynamics.

**Kill condition.** H1 is falsified if, under *all three* pre-registered d_eff definitions (§4), the peak-alignment criterion (§6) fails. Partial support (alignment under some definitions) will be reported as such, with all three definitions shown. **The result will be written up and posted publicly regardless of outcome.**

---

## 2. Design

| Factor | Levels |
|---|---|
| Dataset | CIFAR-10 |
| Architecture | ResNet-18 (setup replicating He et al., ICML 2022) |
| Method | (a) static magnitude prune · (b) static random prune · (c) DST: RigL · (d) DST-mask-retrain |
| Nominal sparsity | 10 log-spaced levels in [0.50, 0.995]; grid may be densified near observed peaks (§7, adaptive rule) |
| Label noise | 0%, 10%, 20% symmetric |
| Seeds | 3 per cell |

**Condition (d) construction:** take the final mask from each condition-(c) run, freeze it, re-initialize weights, train statically. Same seed pairing as parent run.

**Phasing.** Phase 1 (confirmatory core): methods (a) + (c), 10% noise, full sparsity grid, 3 seeds = 60 runs. Phases 2–3 add methods (b), (d) and noise levels {0%, 20%}. Phase 1 alone can trigger the kill condition; later phases cannot rescue H1 if Phase 1 falsifies it.

**Training protocol:** identical optimizer, schedule, epochs, and augmentation across all conditions; hyperparameters fixed at He et al. replication values before Phase 1 and recorded in `configs/`. Models must reach ≤1% train error (interpolation) to enter analysis; non-interpolating runs are reported but excluded from peak estimation.

---

## 3. Measurements (per trained model)

1. Test error, train error (interpolation check)
2. Hessian eigenspectrum at convergence via PyHessian: top-50 eigenvalues (Lanczos), trace (Hutchinson, 100 probe vectors), spectral density
3. All three d_eff definitions (§4)
4. Estimator uncertainty: Hutchinson repeated with 5 independent probe sets → CI per estimate

Hessian computed on the training set (loss landscape the optimizer saw), full network, at final checkpoint.

## 4. Effective-dimension definitions (all three reported, always)

| ID | Definition | Notes |
|---|---|---|
| D1 | N_eff(α) = Σᵢ λᵢ/(λᵢ+α), Maddox et al. 2020 | α = 1/n·(loss curvature scale); α fixed after pilot (§8), before Phase 1 |
| D2 | Spectral-gap count: #{λᵢ > λ_bulk edge} | bulk edge estimated from spectral density |
| D3 | Hessian trace (Hutchinson) | scale proxy; no threshold parameter |

No additional definitions may be introduced post hoc for confirmatory claims.

## 5. Peak estimation

Per (method × noise × definition): fit test-error curve vs. axis variable with local regression (LOESS, span fixed at 0.4); peak = argmax of fit within the sampled range. Peak-location CI via bootstrap over seeds (1000 resamples).

## 6. Confirmatory metric and decision rule

**Collapse ratio** R = (spread of peak locations across methods in log-d_eff coordinates) ÷ (spread in log(1−sparsity) coordinates), spread = max−min across methods.

- **H1 supported** under definition Dk if R < 0.33 and peak-location CIs overlap in d_eff coordinates.
- **H1 falsified** if R ≥ 0.67 under all of D1–D3, or CIs disjoint in d_eff coordinates under all three.
- Intermediate outcomes reported as inconclusive-partial, with all values shown.

Thresholds (0.33 / 0.67) are fixed now, acknowledged as conventional rather than derived.

## 7. Adaptive elements (pre-declared)

Only two adaptations are permitted without a §9 deviation entry:
1. Adding sparsity grid points *between* existing ones to localize a peak (never removing points).
2. Increasing Hutchinson probe count if CI width exceeds 10% of the estimate.

## 8. Pilot phase (pre-freeze)

Before freezing: one throwaway run per method to (a) validate the PyHessian pipeline end-to-end, (b) confirm per-run compute cost, (c) fix α for D1 and the LOESS span. Pilot runs are excluded from confirmatory analysis. The freeze commit happens immediately after, and its hash is recorded here: `[commit hash]`.

## 9. Deviations log

| Date | Change | Reason |
|---|---|---|
| — | — | — |

---

*References: He et al. (ICML 2022); Curth et al. (NeurIPS 2023); Maddox et al. (2020); Evci et al. (ICML 2020); Yao et al., PyHessian (2020).*
