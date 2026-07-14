# AI Makeup Transfer

Extract makeup from a face *with* makeup as a **translucent RGBA layer in 3D UV space**, then
apply it **realistically** to a face *without* makeup (skin tone shows through, so the same look
adapts per person).

> **Status:** early build. Focus is the **NN core** (extract → apply). Storage/sharing is **faked
> with local files** (`./looks/`); a real backend comes later. See `docs/BUILD_PLAN.md`.

## Design in one line
`with-makeup + without-makeup (same person)` → **extract** UV makeup layer (RGBA) → **apply** to any
target face via UV warp + skin-tone-aware blend + harmonization refiner.

## Repo layout
```
docs/            BUILD_PLAN.md, references.md (papers + GitHub repos to implement)
src/makeup_transfer/
  preprocess.py  face detect / landmarks / parsing  (MediaPipe + BiSeNet)
  uv_unwrap.py   image ↔ canonical UV texture (MediaPipe canonical mesh, no 3DMM yet)
  phase0_spike.py  Phase 0 gate: hand-derive a layer + re-apply (PASSED 2026-06-12)
  storage.py     LOCAL-FILE fake storage (swappable interface for future backend)
  extract.py     [Phase 1] pair → makeup_layer.png (RGBA, UV space)   ✅
  removal.py     [Phase 2.5] SSAT makeup removal → pseudo-bare face   ✅ (GPU)
  apply.py       [Phase 2] layer + target → result.png                ✅
  priors.py      [Phase 4] PCA makeup prior: fit instead of difference  ⚠️ opt-in (--prior; gate failed)
  app.py         [Phase 3] Gradio web demo                            ✅
data/samples/    test images
looks/           saved looks (fake storage):  <name>/layer.png + meta.json
third_party/     cloned reference repos (makeup-extract, CPM, 3DDFA_V2, ...)
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the web demo

An interactive Gradio app ties the whole pipeline together: extract a look from a pair (or a
single photo), apply saved looks to new faces with per-region intensity sliders, and browse the
look library. **No model training — all weights are pretrained and already in the repo.**

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # ensures gradio is installed
$env:PYTHONPATH = "src"                   # package lives under src/
python -m makeup_transfer.app
```

Open the printed URL (default http://127.0.0.1:7860). The **Apply** and **Extract-from-a-pair**
tabs run on CPU; the **Extract-from-one-photo** tab uses the SSAT removal GAN and **requires a
CUDA GPU** — install the CUDA torch wheel first if you haven't:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

The app ships seeded with the sample looks under `looks/`; try applying `ffhq-002386-03` to
`data/ffhq_makeup/003417_bare.jpg` to see makeup transfer across faces.

## Build phases (see docs/BUILD_PLAN.md)
- **Phase 0** — feasibility spike: 3D fit → UV, hand-derive one makeup layer, re-apply. Go/no-go.
- **Phase 1** — `extract.py`
- **Phase 2** — `apply.py` (+ fine-tune harmonization refiner)
- **Phase 3** — local save/load + Gradio demo

GPU: developed against an RTX 4060 Ti (8 GB). CPU works for testing, slowly.
