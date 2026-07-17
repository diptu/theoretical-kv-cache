# Experiments — Sparse On-Device Adaptation for Crop Screening Under Drift

Operational breakdown of the frozen plan in [`preregistration.md`](preregistration.md).
Each run is tied to a hypothesis (H1–H4). **Nothing here overrides the pre-registration** —
if a run suggests a change, log it in the pre-registration amendment log, do not edit hypotheses in place.

Legend: ☐ not started · ◐ running · ☑ done · ✗ falsified/negative

---

## Decision gate (read before Phase 1)

The primary metric — recovery ratio `(adapted − frozen) / (oracle − frozen)` — is undefined
until Phase 0 fixes `frozen` and `oracle`. **The `≥ 0.30` H1 threshold and the `k ≤ 25%` H3
budget are locked at the end of Phase 0, before any adaptation result is seen.** Record the
locked values here once, with a commit hash, and never move them.

- Locked thresholds: `recovery_min = ____`  ·  `k_max = ____`  · commit `________`

---

## Phase 0 — Baselines & drift confirmation (prerequisite)

Establishes the two anchors of every later number. If frozen accuracy does **not** drop under
drift, there is nothing to recover and the project stops here.

| ID   | Purpose | Tests | Factors | Status | Result |
| ---- | ------- | ----- | ------- | ------ | ------ |
| E0.1 | Train source models on PlantVillage | — | {MobileNetV2-0.35, MCUNet} × 3 seeds | ☐ | src acc = ____ |
| E0.2 | Measure **frozen** accuracy on each drift target | H1 anchor | backbones × {PlantDoc, Cassava, synthetic} × 3 | ☐ | frozen = ____ |
| E0.3 | Train **cloud oracle** (full labelled retrain on target) | H1 anchor | same grid | ☐ | oracle = ____ |
| E0.4 | Confirm gap `oracle − frozen` is material (≥ 10 pts) on ≥ 2 drift types | go/no-go | — | ☐ | gap = ____ |

**Exit criterion:** E0.4 passes → lock thresholds in the decision gate → proceed. Otherwise, re-scope drift.

Run:
```bash
python -m src.train --backbone mobilenetv2_035 --data plantvillage --seed 0
python -m src.train --eval --ckpt <src> --data plantdoc --mode frozen
python -m src.train --backbone mobilenetv2_035 --data plantdoc --mode oracle --seed 0
```

---

## Phase 1 — Batch adaptation (minimum publishable unit)

All adaptation conditions, unlabelled batch of target data, compared against the Phase 0 anchors.
This is the core H1 test.

Grid: 6 conditions × 2 backbones × 3 drift types × 3 seeds.

| ID   | Condition | Tests | Status | Recovery ratio (mean ± CI) |
| ---- | --------- | ----- | ------ | -------------------------- |
| E1.1 | BN-recal (running-stat recompute) | H1 | ☐ | ____ |
| E1.2 | TENT (affine-only entropy min) | H1 | ☐ | ____ |
| E1.3 | **Sparse on-device update (core)** | H1, H3 | ☐ | ____ |
| E1.4 | Pseudo-label self-training | H1 | ☐ | ____ |

**H1 decision:** computed on the mean recovery ratio of E1.3 across the three drift types
against the locked `recovery_min`. Record pass/fail and commit hash. Do not tune E1.3's
sparse-selection rule on the drift test sets — it is fixed in `configs/sparse_select.yaml`.

Run:
```bash
python -m src.train --adapt sparse_update --ckpt <src> --data plantdoc \
    --tier mcu --config configs/sparse_select.yaml --seed 0
```

---

## Phase 2 — Streaming / continual stability

Feeds the target as a long unlabelled stream instead of a single batch. Tests whether the
core method quietly diverges over time — the classic failure mode of test-time adaptation.

| ID   | Purpose | Tests | Factors | Status | Min acc over stream |
| ---- | ------- | ----- | ------- | ------ | ------------------- |
| E2.1 | Streaming sparse-update, fixed drift target | H2 | best backbone × 3 drift × 3 seeds | ☐ | ____ |
| E2.2 | Streaming under *sequential* drift (target A → B) | H2 | — | ☐ | ____ |

**Kill trigger:** if accuracy at **any** checkpoint falls below the E0.2 frozen baseline,
H1 is falsified per the pre-registration (net-harmful adaptation). Log it and write the negative result.

---

## Phase 3 — Ablations

| ID   | Purpose | Tests | Sweep | Status | Result |
| ---- | ------- | ----- | ----- | ------ | ------ |
| E3.1 | Sparsity efficiency: recovery vs. update budget | H3 | k ∈ {5, 10, 25, 50, 100}% | ☐ | knee at k = ____ |
| E3.2 | Selection rule: top-k grad-mag vs. random vs. BN-only | H3 | 3 rules | ☐ | ____ |
| E3.3 | Abstention gate: selective accuracy vs. coverage | H4 | entropy / energy threshold | ☐ | ____ |
| E3.4 | Synthetic-drift severity curve (uses OCR-style corruption generator) | H1 | 5 severities | ☐ | ____ |

---

## Phase 4 — On-hardware profiling

Numbers must come from a **physical device per tier**, not FLOP estimates, or the edge-budget
claim is unsupported.

| ID   | Tier | Device | Metric | Status | Result |
| ---- | ---- | ------ | ------ | ------ | ------ |
| E4.1 | MCU | (e.g. STM32 / MCU ≤ 256 KB) | peak train RAM, ms/update | ☐ | ____ |
| E4.2 | Cheap-device | Pi Zero 2 W / low-end Android | RAM, latency, mWh/update | ☐ | ____ |

**Budget check:** E1.3's memory footprint must fit the E4.1 tier, else re-report the core
result under the cheap-device tier only and say so plainly.

---

## Results ledger

Single source of truth for the paper's main table. Fill as runs complete.

| Drift target | Frozen | BN-recal | TENT | **Sparse (core)** | Pseudo-label | Oracle |
| ------------ | ------ | -------- | ---- | ----------------- | ------------ | ------ |
| PlantDoc     | ____   | ____     | ____ | ____              | ____         | ____   |
| Cassava      | ____   | ____     | ____ | ____              | ____         | ____   |
| Synthetic    | ____   | ____     | ____ | ____              | ____         | ____   |

---

## Run hygiene

- Every run writes `results/<exp_id>/<seed>/metrics.json` + the exact config used.
- Seeds fixed to {0, 1, 2} everywhere; report mean ± CI, never a single seed.
- One experiment = one commit; put the `exp_id` in the commit message.
- No condition, dataset, budget, or metric added after Phase 1 unblinding.
