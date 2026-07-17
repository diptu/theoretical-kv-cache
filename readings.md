# Readings — Sparse On-Device Adaptation for Crop Screening Under Drift

Curated, annotated, and ordered — not a flat dump. Each entry says *why it matters for this
project*. Read for what you need to **use**, not to re-derive.

**Legend**  ★ = read first / essential · difficulty 🟢 accessible · 🟡 moderate · 🔴 math-heavy (skim for intuition, don't get stuck on proofs)

**Suggested path (≈4 weeks):**
Week 1 — Orientation + Motivation. Week 2 — On-device & sparsity. Week 3 — Label-free adaptation. Week 4 — Drift/robustness + crop data. Extensions and foundations as needed alongside.

---

## 0. Motivation — the paper that justifies the whole project

- ★ **Mohanty, Hughes & Salathé (2016)** — *Using Deep Learning for Image-Based Plant Disease Detection*, Frontiers in Plant Science. 🟢
  The canonical result: ~99% accuracy on clean PlantVillage images collapses to ~31% on real field photos. **This lab→field collapse is exactly the drift your project attacks.** Cite it in your first paragraph.

## 1. Orientation (the big picture)

- ★ **Warden & Situnayake (2019)** — *TinyML*, O'Reilly. 🟢
  Practical grounding for what "runs on a microcontroller" actually means. Read the concepts, skip nothing about memory budgets.
- **"Intelligence at the Extreme Edge: A Survey on Reformable TinyML"** (ACM Computing Surveys, 2023). 🟡
  Frames *adaptive/reformable* TinyML — your niche. Good map of what's done vs. open.
- ★ **Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia (2014)** — *A Survey on Concept Drift Adaptation*, ACM Computing Surveys. 🟡
  The vocabulary of drift (covariate, prior, concept). You'll use these terms precisely in the preregistration.

## 2. The devices — TinyML & on-device training (the "cheap, offline" pillar)

- ★ **Lin, Zhu, Chen, Gan & Han (2022)** — *On-Device Training Under 256KB Memory*, NeurIPS. 🟡🔴
  The keystone. Training fits in 256KB only by updating a **sparse** subset of parameters. Your core method builds directly on this.
- **Lin et al. (2020)** — *MCUNet: Tiny Deep Learning on IoT Devices*, NeurIPS. 🟡
  How to get a usable model onto an MCU in the first place (your backbone tier).
- **Sandler et al. (2018)** — *MobileNetV2*, CVPR. 🟢
  Your other backbone; understand inverted residuals + why it's edge-friendly.
- **Jacob et al. (2018)** — *Quantization and Training of NN for Integer-Arithmetic-Only Inference*, CVPR. 🟡
  int8 quantisation — how your models actually shrink for deployment.

## 3. Adapting without labels (the "keeps adapting" pillar)

- ★ **Wang, Shelhamer, Liu, Olshausen & Darrell (2021)** — *Tent: Fully Test-Time Adaptation by Entropy Minimization*, ICLR. 🟡
  Your primary baseline; adapts using only unlabelled test data (BN affine params). Understand it cold.
- **Liang, Hu & Feng (2020)** — *SHOT: Do We Really Need to Access the Source Data?*, ICML. 🟡
  Source-free adaptation — matches your "no cloud, no source data on device" constraint.
- **Sun et al. (2020)** — *Test-Time Training with Self-Supervision*, ICML. 🟡
  The self-supervised objective idea behind label-free updates.
- ★ **Wang, Fink, Van Gool & Dai (2022)** — *Continual Test-Time Domain Adaptation (CoTTA)*, CVPR. 🟡
  Directly relevant to your **Phase 2 streaming** runs and the divergence failure mode.

## 4. Sparsity & compression (bridge from your prior work)

- **Frankle & Carbin (2019)** — *The Lottery Ticket Hypothesis*, ICLR. 🟡
  Why sparse subnetworks can match dense ones — the premise behind sparse updates.
- **Evci et al. (2020)** — *RigL: Making All Tickets Winners*, ICML. 🟡
  Dynamic sparse training — *which* weights to keep/update. This is the direct link to your old repo; your sparse-selection rule descends from here.
- **Han et al. (2016)** — *Deep Compression*, ICLR. 🟢
  Pruning + quantisation + coding — the classic compression pipeline.
- *Your own frozen SDD study* (`sdd-double-descent` branch) — the effective-dimension lens on sparsity is worth citing as your prior methodology.

## 5. The drift problem — distribution shift & robustness

- ★ **Hendrycks & Dietterich (2019)** — *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR. 🟢
  The corruption suite. **Your synthetic-drift generator (Phase 3, E3.4) builds on this** — and it ties straight back to your OCR synthetic-noise paper.
- **Koh et al. (2021)** — *WILDS: A Benchmark of in-the-Wild Distribution Shifts*, ICML. 🟡
  How the field measures real-world shift rigorously; useful for framing your evaluation.

## 6. Continual learning & failure modes (stability)

- **Kirkpatrick et al. (2017)** — *Overcoming Catastrophic Forgetting (EWC)*, PNAS. 🟡
  Why on-device adaptation can erase what the model knew — the risk behind your H2 stability kill-trigger.

## 7. Crop data & the domain

- ★ **Hughes & Salathé (2015)** — *An open access repository of images on plant health (PlantVillage)*. 🟢
  Your source dataset (54,303 images, 38 classes). Note: **lab conditions** — that's the point.
- ★ **Singh, Jain, Jain, Kayal, Kumawat & Batra (2020)** — *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD (DOI 10.1145/3371158.3371196). 🟢
  Your field/drift target (2,598 real internet images, 13 species, up to 17 classes). GitHub: `pratikkayal/PlantDoc-Dataset`.
- ★ **Cassava Leaf Disease** — iCassava (Makerere, FGVC/CVPR 2019 workshop) → Kaggle 2020 (~21,400 field images, 4 diseases). 🟢
  Second real-field drift target; strong "global South smallholder" relevance.
- **"Plant disease recognition datasets in the age of deep learning: challenges and opportunities"** (2024 survey). 🟢
  Landscape of what data exists and where the gaps are.

## 8. Foundations to shore up (only what you'll use)

You don't need to re-derive theory — you need enough to read the papers above fluently. Target *fluency in use*, not proofs.

- **d2l.ai — Dive into Deep Learning** (Zhang et al.), interactive book. 🟢 — practical PyTorch-first refresher; best ROI for your background.
- **Goodfellow, Bengio & Courville — Deep Learning**, Ch. 2–5 only. 🟡 — linear algebra / probability / numerical bits you'll actually touch (BatchNorm stats, entropy, gradients). Skip the rest for now.

## 9. Optional / extension reading

- **McMahan et al. (2017)** — *Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)*, AISTATS. 🟡 — for a federated-across-microgrids extension chapter.
- **Hinton, Vinyals & Dean (2015)** — *Distilling the Knowledge in a Neural Network*. 🟢 — an alternate route to tiny models.

---

## Notes to self

- Keep a one-paragraph summary per ★ paper in `analysis/notes/` — future SOP and related-work section come straight from these.
- For each method paper, note: what it assumes about compute/labels, and whether that assumption survives *your* offline, unlabelled, MCU setting. The papers whose assumptions break are your contribution.
