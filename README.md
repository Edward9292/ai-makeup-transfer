# AI Makeup Transfer

Take **any photo of someone wearing makeup** and put that makeup on **any other face**,
realistically: the wearer keeps their own skin texture, lighting and complexion; only the
makeup travels.

> **Status:** reference-based engine live (2026-11-13) — see "Design" below for what changed and
> why. Site: **https://makeup-transfer.edwardlin.dev** · live demo: Hugging Face Space (CPU) ·
> hosting details: `docs/HOSTING.md`. Storage/sharing is still **faked with local files**
> (`./looks/`); a real backend is a documented upgrade path, not built. See `docs/BUILD_PLAN.md`.

## Design in one line
`reference photo (anyone, with makeup) + target photo` → **CSD-MT** transfer network renders the
makeup onto a 256 px face crop → the **makeup delta** is composited back onto the full-resolution
photo, gated by face parsing + a skin-visibility mask, with per-region strengths and a
*complexion* control that keeps the target's own skin tone.

**Why the rewrite.** The first design extracted makeup as an RGBA layer in UV space by differencing
a with/without pair of the *same* person, then blended that layer onto targets. In practice the
difference matte kept only the strongest signal (a lip tint) and threw eye shadow, liner and
blush away as "illumination", so applied looks read as a flat color wash. A network trained on
makeup/no-makeup pairs is far better at deciding what is makeup; we keep our own full-resolution
compositing because the network's 256 px output alone would soften the photo.

## Repo layout
```
docs/            BUILD_PLAN.md, HOSTING.md, references.md
site/            static Next.js showcase site (Vercel root dir = site/)
scripts/         build_space.py — stages the Hugging Face Space demo into space_build/
src/makeup_transfer/
  engine.py      ⭐ reference-based transfer: crop → CSD-MT → full-res delta composite
  vendor/csd_mt/ vendored CSD-MT generator + BiSeNet parser (inference code only)
  fetch_weights.py  downloads the engine's two .pth files into weights/
  preprocess.py  face detect / landmarks / region masks   (MediaPipe)
  occlusion.py   skin vs. hair/hands/glasses mask          (MediaPipe segmenter)
  storage.py     LOCAL-FILE fake storage (swappable interface for future backend)
  apply.py       look + target → result.png (engine looks; legacy UV-layer looks still work)
  app.py         Gradio web demo
  uv_unwrap.py, extract.py, removal.py, priors.py   legacy UV-layer pipeline (kept, not default)
weights/         MediaPipe assets (tracked) + engine weights csd_mt.pth / bisenet_79999_iter.pth (fetched)
data/ffhq_makeup/  FFHQ-Makeup sample pairs (CC BY-NC-SA, see Licenses below)
looks/           saved looks: <name>/reference.png + meta.json (3 samples tracked)
third_party/     NOT tracked — cloned on demand for the legacy GPU single-photo path
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt            # includes torch (CPU is fine)
python -m makeup_transfer.fetch_weights    # ~80 MB: CSD-MT generator + BiSeNet parser
```

Set `MAKEUP_DEVICE=mps` (Apple) or `cuda` to run the network on a GPU; CPU takes 1–4 s per
transfer on a laptop, which is what the hosted demo uses.

The MediaPipe assets under `weights/` are tracked, so setup is just the pip install. If one is
ever missing, re-fetch it with:

```powershell
curl -L -o weights/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o weights/selfie_multiclass_256x256.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
```

The segmenter (`selfie_multiclass_256x256.tflite`) is what keeps hair, hands and glasses out of
an extracted look and stops makeup being painted on top of them at apply time. Without it the
pipeline still runs — occlusion masking just switches itself off.

## Run the web demo

```powershell
$env:PYTHONPATH = "src"
python -m makeup_transfer.app
```

Open the printed URL (default http://127.0.0.1:7860). The demo is three steps in the order you
work in them — **Try on** (pick a look + a face → before/after, with lips / eyes / brows / skin
and *complexion* sliders), **Create a look** (one photo of anyone wearing makeup → saved look),
**My looks** (browse / export / delete). All three run on CPU; try `ffhq-002386-03` on
`data/ffhq_makeup/003417_bare.jpg`.

### Command line

```powershell
# any photo with makeup → any other photo, in one go
python -m makeup_transfer.engine data/ffhq_makeup/003417_bare.jpg data/ffhq_makeup/002386_makeup_03.jpg `
    --out preprocess_out/engine --strength lips=1.2 --complexion 0.3
# or apply a saved look
python -m makeup_transfer.apply ffhq-002386-03 data/ffhq_makeup/003417_bare.jpg
```

### Legacy UV-layer pipeline

`extract.py` / `uv_unwrap.py` / `priors.py` / `removal.py` are the previous pair-difference
design (see `docs/BUILD_PLAN.md` for its history and why it was replaced). Looks saved by it
(`layer.png`) still apply through `apply_look`; the SSAT single-photo path still needs CUDA and
the untracked `third_party/` clones described in BUILD_PLAN.

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
- **MediaPipe** assets in `weights/` (face landmarker, selfie multiclass segmenter, canonical
  face model): Apache-2.0, © Google.
- **CSD-MT** (the makeup engine; vendored inference code in `src/makeup_transfer/vendor/csd_mt/`
  and the fetched `weights/csd_mt.pth`): **CC BY-NC-SA 4.0** — © Zhaoyang Sun et al.,
  [Snowfallingplum/CSD-MT](https://github.com/Snowfallingplum/CSD-MT) (CVPR 2024). Non-commercial.
- **BiSeNet face parsing** (`weights/bisenet_79999_iter.pth`, vendored model code): MIT —
  [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch).
- **Vendored on demand** (not tracked, legacy path only): [SSAT](https://github.com/Snowfallingplum/SSAT)
  (AAAI 2022).
- **Research this build stands on:** CSD-MT, makeup-extract, Makeup Priors (CVPR 2024), CPM,
  Stable-Makeup, FLUX-Makeup — see `docs/references.md`.

Developed on an Apple M2 (CPU) and an RTX 4060 Ti (8 GB). The engine runs on CPU everywhere;
only the legacy SSAT removal needs CUDA.
