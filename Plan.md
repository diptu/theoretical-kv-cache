# Implementation Plan: KV-Cache-Focused LLM Visualization

Goal: build an interactive 3D "live paper" in the spirit of
[bbycroft.net/llm](https://bbycroft.net/llm) (source: [bbycroft/llm-viz](https://github.com/bbycroft/llm-viz)),
but scoped to *this repo's* actual research focus — KV cache structure, redundancy, and
compression — rather than a full GPT-2/GPT-3-scale clone. It becomes the flagship
interactive piece of the "Interactive Research Platform" described in `Requirements.md`.

## Reference: what bbycroft's site actually does (and what we're adapting)

Researched directly from the live site + source repo, not assumed:

- **Stack:** Next.js 13 (Pages Router) + React 18 + TypeScript, Tailwind + SCSS modules,
  `d3-color` for colormaps, FontAwesome for icons. **No Three.js** — rendering is a
  hand-written WebGL2 engine (`src/llm/render/{blockRender,lineRender,fontRender,
  triRender,threadRender}.ts`) with MSDF-font text rendered directly in the 3D scene.
- **It visualizes real tensors, not an abstract graph.** Every intermediate value
  (token embeddings, Q/K/V, attention-score grids, MLP activations, logits) is drawn as
  a 3D block of value-colored cells sized to the tensor's actual dimensions — this is
  the core idea, and the one thing our original draft plan got wrong (it modeled
  generic "neurons as spheres / edges as cylinders").
- **The numbers are real.** A GPT-style forward pass runs via a WASM inference engine
  (`GptModelWasm.ts`) so hovering a cell shows an actual computed value, not decoration.
- **Guided walkthrough.** A scripted tour (`walkthrough/`, `Commentary.tsx`,
  `PhaseTimeline.tsx`, `Annotations.ts`) drives the camera and narration step-by-step
  through the computation graph.

### Deliberate deviations from that stack (and why)

| Area | bbycroft/llm-viz | This project | Why |
|---|---|---|---|
| Renderer | Custom WebGL2 engine | React Three Fiber + drei | Hand-rolled WebGL is a multi-month investment; R3F gives instancing, raycasting, and camera controls for free at the scale we need (single/few nano-scale models, not GPT-3). |
| Inference | WASM-compiled C inference engine, in-browser, arbitrary input | Python/FastAPI backend (Phase 0, implemented) serving live traces for arbitrary in-vocabulary input, plus precomputed JSON for the fixed example set; in-browser WASM/JS inference remains a stretch goal | Solo project with limited bandwidth (see README: currently in GRE/IELTS foundational phase). A server-side backend was far faster to stand up than a WASM engine and already gives *real* numbers for arbitrary input, not just the fixed examples. |
| Scope | General GPT-2/GPT-3 architecture tour | Attention + KV cache lifecycle specifically (growth across decode steps, eviction/compression strategies from the paper list in `README.md`: StreamingLLM, H2O, SnapKV, KIVI) | This is the repo's actual research subject — a narrower, deeper tool beats a shallow full clone. |

## Phase 0: Model & Data Foundation (Python) — *implemented, see `backend/`*

The original plan jumped straight to "convert model weights to coordinate arrays"
without ever computing real activations at runtime. This phase produces the actual
numbers the visualization will render.

- [x] Train (or reuse) a tiny char-level nanoGPT-style model (2–4 layers, small
      `d_model`) on a toy task — mirror Karpathy's minGPT sorting/Shakespeare-char
      approach that bbycroft's demo model itself uses. →
      `backend/app/model.py` (3-layer/3-head/48-dim GPT) +
      `backend/app/dataset.py` (digit-sort task) + `backend/app/train.py`
      (100% sort accuracy well within 3000 steps, CPU, under a minute).
- [x] Instrument the forward pass to dump, per token position and per layer:
      embeddings, Q/K/V, post-softmax attention pattern, attention output, MLP
      activations, logits — **and the KV cache tensors at each autoregressive decode
      step**. → `backend/app/model.py` (`collect_trace`/`past_kvs`) +
      `backend/app/tracing.py`.
- [x] Export as one JSON/binary "trace" per example prompt (`Float32Array`-friendly
      layout), covering a handful of fixed example inputs to start. →
      `backend/app/export_traces.py` → `backend/traces/example_*.json`; also
      served live via `POST /trace` (`backend/app/main.py`, FastAPI) for
      arbitrary in-vocabulary token sequences, not just the fixed examples.
- [x] Keep the export format versioned/documented so the frontend's `DataService`
      (Phase 3) has a stable contract. → `version: "1.0"`, documented in
      `backend/README.md`.

## Phase 1: Infrastructure & Environment Setup

- [ ] Initialize Next.js 15+ (App Router, TypeScript).
- [ ] Install: `three`, `@types/three`, `@react-three/fiber`, `@react-three/drei`, `zustand`.
- [ ] Configure `next.config.js` for Three.js/Drei ESM transpilation.
- [ ] `Canvas` wrapper component with `ssr: false` for correct hydration.

## Phase 2: Core Rendering Primitives — *replaces generic "NeuralNetwork" concept*

- [ ] `TensorBlock` component: a 3D grid of cells (`InstancedMesh` cubes) representing
      one tensor, with a custom shader material mapping value → color via a shared
      colormap (port `d3-color`'s interpolation approach).
- [ ] `FlowConnector` component: lines/arrows between `TensorBlock`s showing data flow
      (embedding → Q/K/V → attention → MLP → logits).
- [ ] In-scene text labels via drei `<Text>` (SDF-based, same idea as bbycroft's
      msdf-bmfont approach) for tensor names/dimensions.
- [ ] Scene chrome: `PerspectiveCamera`, ambient + directional lighting, `OrbitControls`.

## Phase 3: Data Pipeline

- [ ] `DataService`: fetch and parse the Phase 0 trace files into `Float32Array`s.
- [ ] `WebWorker` layer to parse trace data off the main thread.
- [ ] Lazy-load traces per example so initial bundle stays small.

## Phase 4: Model Scene Assembly

- [ ] Lay out the real GPT computational graph in 3D: embedding table → stacked
      transformer blocks (each showing Q/K/V blocks, attention pattern grid, output
      projection, MLP) → final layer norm → logits — following the same conceptual
      layout as bbycroft's `GptModelLayout.ts`, not a generic layered network.
- [ ] Drive block dimensions and positions from the trace metadata (Phase 0/3), not
      hardcoded coordinates.

## Phase 5: Walkthrough / Narrative Engine

- [ ] Declarative walkthrough steps array: `{ cameraTarget, focusedTensor, commentaryMarkdown }`.
- [ ] `Commentary` panel + `PhaseTimeline` scrubber UI, mirroring bbycroft's
      `Commentary.tsx` / `PhaseTimeline.tsx`.
- [ ] Camera `lerp`/easing between steps driven by the Zustand store.

## Phase 6: KV Cache Feature — *the differentiator vs. a plain clone*

- [ ] Autoregressive decode-step slider: show the KV cache tensor growing one token at
      a time.
- [ ] Toggleable eviction/compression overlays illustrating the strategies studied in
      this repo (StreamingLLM sliding window, H2O, SnapKV, KIVI quantization) — visually
      mark which cache slots are kept vs. evicted/compressed at each step.
- [ ] Redundancy view: highlight low-rank/near-duplicate structure across cached K/V
      vectors (ties directly to the "KV Cache redundancy" research question in `README.md`).

## Phase 7: State Management & Interaction

- [ ] `useStore` (Zustand): `activeLayer`, `decodeStep`, `hoveredCell`, `selectedCell`,
      `walkthroughStepIndex`, `evictionStrategy`.
- [ ] Hover/click a cell → highlight the contributing cells in the previous tensor
      (dependency trace) + tooltip with the actual numeric value, mirroring
      `Interaction.ts`/`Annotations.ts`.
- [ ] Sidebar UI for model documentation and per-layer metrics.

## Phase 8: Content Integration

- [ ] Wire this visualization into the broader MDX "Research Narrative" engine from
      `Requirements.md` (KaTeX derivations linking to the interactive tensors it explains).

## Phase 9: Optimization & Polish

- [ ] Frustum culling / LOD only if a real model size (GPT-2-scale) is attempted later.
- [ ] Bloom / post-processing via `@react-three/postprocessing` as a final visual pass.

## Phase 10: Deployment

- [ ] Optimize bundle size; verify trace JSON/binary assets are chunked per example.
- [ ] Deploy via Vercel (matches `Requirements.md`'s deployment plan).

## Stretch Goal: Live Inference

- [x] Server-side live inference for arbitrary in-vocabulary token sequences via the
      Phase 0 FastAPI backend (`POST /trace`, not just the fixed example set) — already
      available, see `backend/README.md`.
- [ ] In-browser inference (no Python backend dependency) for arbitrary user input, via
      a WASM-compiled forward pass (own engine, or `onnxruntime-web` loading an exported
      nanoGPT/GPT-2-small checkpoint) — only after the precomputed-trace MVP above is
      working end-to-end. The current FastAPI `/trace` endpoint is a reasonable
      intermediate step: the frontend can already hit it directly during development
      before an in-browser engine exists.
