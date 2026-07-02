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
  extract.py     [Phase 1] pair → makeup_layer.png (RGBA, UV space)   (TODO)
  apply.py       [Phase 2] layer + target → result.png                (TODO)
  app.py         [Phase 3] Gradio demo                                 (TODO)
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

## Build phases (see docs/BUILD_PLAN.md)
- **Phase 0** — feasibility spike: 3D fit → UV, hand-derive one makeup layer, re-apply. Go/no-go.
- **Phase 1** — `extract.py`
- **Phase 2** — `apply.py` (+ fine-tune harmonization refiner)
- **Phase 3** — local save/load + Gradio demo

GPU: developed against an RTX 4060 Ti (8 GB). CPU works for testing, slowly.
