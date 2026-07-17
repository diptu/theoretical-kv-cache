```text
                        .-. |\/| .-.           a tiny model,
                         '.`|  |'.'            grown on silicon,
                      .----( )( )----.         that keeps learning
                     |  ::  ....  ::  |        in the field — offline
                     |  ::  ::::  ::  |
                      '--.--.--.--.--'
                         |  |  |  |

     ____ __  __ _ _ _ ___ ___ ___ ___ __| |__ _ ___
    (_-< '_ \/ _` | '_(_-</ -_)___/ -_) _` / _` / -_)
    /__/ .__/\__,_|_| /__/\___|   \___\__,_\__, \___|
       |_|                                 |___/
       __ _ _ ___ _ __ ___ __ _ __| |__ _ _ __| |_
      / _| '_/ _ \ '_ \___/ _` / _` / _` | '_ \  _|
      \__|_| \___/ .__/   \__,_\__,_\__,_| .__/\__|
                 |_|                      |_|

    label-free · sparse on-device adaptation · crop screening under drift
    ─────────────────────────────────────────────────────────────────────

```
# Sparse On-Device Adaptation for Crop Screening Under Drift

`#tinyml` · `#on-device-training` · `#test-time-adaptation` · `#precision-agriculture`

**Can label-free, *sparse* on-device adaptation keep a compressed crop-disease classifier accurate under field drift — within an edge memory budget — without ever touching the cloud?**

A pre-registered experiment. Most TinyML-for-agriculture work ships a model that is trained once in the cloud and then **frozen** on the device. In a low-connectivity smallholder setting that assumption breaks: the field distribution drifts (new lighting, cheap cameras, cluttered backgrounds, new seasons and crop varieties), there is no connectivity to retrain, and there are no local labels. The frozen model silently degrades and no one notices.

This project tests whether a tiny model can **adapt itself on-device, label-free, by updating only a sparse subset of its parameters** — recovering accuracy lost to drift at a cost that fits an edge budget, without destabilising over time.

## Hypothesis

A compressed CNN classifier trained on clean (lab) crop imagery loses significant accuracy under field and seasonal drift. **Label-free, sparse on-device updates** (self-supervised, touching only a small selected parameter subset) recover a meaningful fraction of the frozen→oracle accuracy gap, at an update cost that fits a defined edge budget, and *without* collapsing below the frozen baseline over a long drift stream.

**Why sparse:** full on-device backprop does not fit a microcontroller's memory. Recent work shows training under 256 KB is possible only by updating a sparse subset of parameters (Lin et al., 2022). *Which* subset to update is a sparse-selection problem — the direct link to this repo's prior sparse-training work.

**Kill condition (pre-registered):** if, across **all** pre-registered drift types, label-free sparse adaptation recovers **< 30%** of the frozen→oracle gap, **or** drops **below** the frozen baseline at any point in the streaming protocol (net-harmful adaptation), the central hypothesis is falsified. The negative result is published either way — "don't rely on test-time adaptation at the edge" is itself a useful finding.

![drift adaptation](DST.jpeg)

## Design

|               |                                                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Source (train)| PlantVillage (clean / lab conditions)                                                                                       |
| Drift targets | PlantDoc (real field) · Cassava Leaf Disease (real field) · synthetic corruptions (controlled covariate shift)             |
| Backbone      | MobileNetV2-0.35 and MCUNet, int8-quantised                                                                                 |
| Edge budgets  | **MCU tier:** ≤ 256 KB training memory (sparse-update only) · **Cheap-device tier:** Raspberry Pi Zero 2 W / low-end phone  |
| Conditions    | Frozen (lower bound) · BN-stats recalibration · TENT (affine-only entropy min) · **Sparse on-device update (core)** · pseudo-label self-training · Cloud oracle (full labeled retrain, upper bound) |
| Protocol      | Batch adaptation **and** streaming/continual adaptation (tests stability over a long drift sequence)                        |
| Seeds         | 3                                                                                                                            |

**Synthetic corruptions** reuse the synthetic-data-under-noise approach from prior work (OCR, 2020) to generate *controlled, labelled* drift — isolating covariate shift from confounds the field datasets carry.

## Metrics

- **Accuracy under drift** — frozen vs. each adaptation method vs. oracle.
- **Recovery ratio** — `(adapted − frozen) / (oracle − frozen)`; the headline number.
- **Adaptation cost** — extra RAM, FLOPs/update, latency, and estimated energy per update, *measured on the target tier* (not simulated).
- **Stability** — accuracy trajectory over the streaming protocol; does it avoid catastrophic collapse?
- **Calibration & abstention** — ECE, and quality of an "unsure / out-of-distribution → refer to human" gate.

## Status

- [ ] Literature-review gap note (frozen-model assumption; sparse on-device adaptation for drift is open)
- [ ] Pre-registration frozen (`preregistration.md`)
- [ ] Frozen-baseline degradation replicated (PlantVillage → PlantDoc / Cassava)
- [ ] Sparse-update mechanism + memory/energy profiling harness
- [ ] Phase 1: batch adaptation across all conditions
- [ ] Phase 2: streaming/continual stability runs
- [ ] Preprint

## Repository

```
├── preregistration.md   # frozen hypotheses, conditions, kill condition
├── src/
│   ├── data/            # dataset loaders, drift splits, synthetic corruptions
│   ├── models/          # backbones, quantisation, int8 export
│   ├── adapt/           # BN-recal, TENT, sparse on-device update, pseudo-label
│   ├── profile/         # memory / FLOPs / latency / energy measurement
│   └── train.py         # source training + evaluation entry point
├── configs/             # per-condition run configs
├── results/             # per-run metrics + trajectories
└── analysis/            # recovery-ratio and stability plots
```

## Key references

Lin et al., *On-Device Training Under 256KB Memory* (NeurIPS 2022) · Wang et al., *TENT: Fully Test-Time Adaptation by Entropy Minimization* (ICLR 2021) · Evci et al., *RigL* (ICML 2020) · Lin et al., *MCUNet* (NeurIPS 2020) · Hughes & Salathé, *PlantVillage* (2015) · Singh et al., *PlantDoc* (2020) · *Cassava Leaf Disease Classification* (Makerere / Kaggle, 2020) · Gama et al., *A Survey on Concept Drift Adaptation* (2014)

## License

MIT
