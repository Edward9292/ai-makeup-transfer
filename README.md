# AI Makeup Transfer

Extract makeup from a face *with* makeup as a **translucent RGBA layer in 3D UV space**, then
apply it **realistically** to a face *without* makeup (skin tone shows through, so the same look
adapts per person).

> **Status:** NN core complete (extract → apply → demo), packaged for the web.
> Site: **https://makeup-transfer.edwardlin.dev** · live demo: Hugging Face Space (CPU) ·
> hosting details: `docs/HOSTING.md`. Storage/sharing is still **faked with local files**
> (`./looks/`); a real backend is a documented upgrade path, not built. See `docs/BUILD_PLAN.md`.

## Design in one line
`with-makeup + without-makeup (same person)` → **extract** UV makeup layer (RGBA) → **apply** to any
target face via UV warp + skin-tone-aware blend + delta composite.

## Repo layout
```
docs/            BUILD_PLAN.md, HOSTING.md, references.md
site/            static Next.js showcase site (Vercel root dir = site/)
scripts/         build_space.py — stages the Hugging Face Space demo into space_build/
src/makeup_transfer/
  preprocess.py  face detect / landmarks / region masks   (MediaPipe)
  uv_unwrap.py   image ↔ canonical UV texture (MediaPipe canonical mesh, no 3DMM)
  storage.py     LOCAL-FILE fake storage (swappable interface for future backend)
  extract.py     [Phase 1]   pair → makeup_layer.png (RGBA, UV space)      ✅
  removal.py     [Phase 2.5] SSAT makeup removal → pseudo-bare face        ✅ (GPU)
  apply.py       [Phase 2]   layer + target → result.png                   ✅
  priors.py      [Phase 4]   PCA makeup prior  ⚠️ opt-in (--prior; gate failed, see BUILD_PLAN)
  app.py         [Phase 3]   Gradio web demo                               ✅
weights/         MediaPipe landmarker + canonical mesh + fitted PCA prior (tracked, small)
data/ffhq_makeup/  FFHQ-Makeup sample pairs (CC BY-NC-SA, see Licenses below)
looks/           saved looks: <name>/layer.png + meta.json (3 samples tracked)
third_party/     NOT tracked — cloned on demand for the GPU single-photo path (see below)
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Torch is only needed for the **single-photo (GPU) path**; everything else — pair extraction,
apply, the web demo's CPU tabs — runs without it.

## Run the web demo

```powershell
$env:PYTHONPATH = "src"
python -m makeup_transfer.app
```

Open the printed URL (default http://127.0.0.1:7860). The **Extract-from-a-pair**, **Apply**,
and **Look library** tabs run on CPU with the assets already in this repo; try applying
`ffhq-002386-03` to `data/ffhq_makeup/003417_bare.jpg`.

### Optional: the single-photo (GPU) tab

Needs a CUDA GPU plus two untracked third-party pieces (the tab is disabled with a note
until they exist):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/Snowfallingplum/SSAT third_party/SSAT          # weights ship in-repo
git clone https://github.com/zllrunning/face-parsing.PyTorch third_party/face-parsing.PyTorch
# BiSeNet weights (79999_iter.pth) → third_party/face-parsing.PyTorch/res/cp/
#   (Google Drive link in that repo's README; or: gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812)
```

## Website + hosted demo

- `site/` — static Next.js showcase (`cd site && npm install && npm run build`), deployed on
  Vercel at makeup-transfer.edwardlin.dev.
- `scripts/build_space.py` — stages the torch-free Gradio demo into `space_build/` for a free
  Hugging Face CPU Space.
- Full hosting plan, tier limits, deploy runbook, upgrade path: **`docs/HOSTING.md`**.

## Licenses & credits

- **Code** (this repo): MIT — see `LICENSE`.
- **FFHQ-Makeup** sample images in `data/ffhq_makeup/`, the seed looks in `looks/`, and the
  fitted prior `weights/makeup_prior_pca.npz` (derived from FFHQ-Makeup): **CC BY-NC-SA** —
  © the [FFHQ-Makeup](https://huggingface.co/datasets/cyberagent/FFHQ-Makeup) authors
  (CyberAgent). Non-commercial use with attribution; derivatives keep this license.
- **MediaPipe** assets in `weights/` (face landmarker, canonical face model): Apache-2.0, © Google.
- **Vendored on demand** (not tracked): [SSAT](https://github.com/Snowfallingplum/SSAT) (AAAI
  2022), [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (MIT).
- **Research this build stands on:** makeup-extract, Makeup Priors (CVPR 2024), CPM, MT dataset
  (research-scoped samples) — see `docs/references.md`.

GPU: developed against an RTX 4060 Ti (8 GB). CPU works for everything except SSAT removal.
