# Build Plan: AI Makeup Transfer — NN Core First (local, no server)

## Context

Goal (eventual): a web app that **extracts makeup** from a face *with* makeup and **applies it
realistically** to a face *without* makeup, with save/share of looks.

**Current scope (deliberately narrowed):** build the **actual NN pipeline** (the primary feature)
and prove it works locally. **No cloud, no backend, no database** — storage is **faked with local
files**. The real server/storage comes later.

### Confirmed design decisions
- Makeup = a **translucent RGBA layer** (extract as a "transparent PNG"), re-applied with a
  **skin-tone-aware blend** so it adapts to different skin and looks real (not an opaque paste).
- Layer lives in **3D UV texture space** (robust to pose / face shape).
- **Pair-first** extraction: input = with-makeup + without-makeup of the **same person**.
- Build by **reusing pretrained models** — NOT training a makeup GAN from scratch.
- Validated by research: `makeup-extract` (bare skin + makeup + alpha in UV), CPM, SOGAN.
  See `references.md` for papers + repos.

## Scope

**In scope now:** extraction NN, application NN, run on local sample images, save outputs to a
local folder, a thin local demo to view results.

**Deferred:** cloud storage, Postgres, job queues, auth, sharing service, production frontend,
live AR/video, single-image extraction (makeup removal).

## Pipeline (the primary feature)

```
WITH-MAKEUP  ┐
             ├─► preprocess ─► 3D fit → UV ─► [Extraction NN] ─► makeup_layer.png (RGBA, UV space)
WITHOUT      ┘                                                          │  saved to ./looks/<name>/layer.png
                                                                        ▼
TARGET ─► preprocess ─► 3D fit → UV ─► [Apply NN: warp + blend + refine] ─► result.png
```

### Components (each independently runnable)
1. **Preprocess** — face detect, landmarks, parsing mask. Reuse **MediaPipe Face Mesh** + **BiSeNet**.
2. **3D fit → UV** — fit 3DMM, render canonical UV texture. Reuse **3DDFA_V2** or **DECA**.
3. **Extraction NN (pair)** — illumination-aware difference of with/without UV textures → makeup
   **RGBA layer**. Primary blueprint: **makeup-priors** (CVPR 2024, same authors as makeup-extract:
   fit a learned PCA/StyleGAN2 makeup prior in UV space instead of raw differencing — regularizes
   away shadows/noise, faster). makeup-extract = the concept; makeup-priors = the method.
   Keep **CPM**'s separate pattern branch idea for high-freq content (eyeliner, glitter).
4. **Application NN** — warp UV layer to target UV → translucent **blend** (multiply/soft-light,
   skin shows through) → small **harmonization U-Net** for realism → re-project UV→image.
   The refiner is the **only** piece to fine-tune; rest is pretrained/classical.

## Faked storage (local only)
- A **look** = folder `./looks/<look_name>/` with `layer.png` (RGBA) + `meta.json`
  (source filenames, blend params). Stands in for the future DB/object store.
- Save = write folder. Load/apply = read `layer.png`. Share = copy/zip the folder for now.
- Keep all storage calls behind a tiny `storage.py` interface so the cloud backend can replace
  it later without touching the NN code.

## Tech stack (minimal, local)
- **Python + PyTorch**; **MediaPipe**, **BiSeNet**, **3DDFA_V2/DECA**, **CPM/SSAT** weights.
- GPU recommended (local CUDA, or Colab/RunPod for dev). CPU works for testing, slowly.
- **Demo UI:** single-file **Gradio** (or Streamlit) app — upload images, view results.

## Step-by-step

**Phase 0 — Feasibility spike (go/no-go).**
- Run **makeup-extract** and/or **CPM** + **3DDFA_V2/DECA** on samples; produce a UV texture.
- Hand-derive a makeup RGBA layer from one with/without pair (UV difference); re-apply to a third
  face with a multiply blend. Eyeball realism. **Gate: does the layer concept look plausible?**

**Phase 1 — Extraction pipeline (`extract.py`).**
- preprocess → UV → makeup `layer.png` (RGBA). Clean alpha matte; suppress shadows via UV symmetry.
- **Illumination handling (the #1 risk — our UV unwrap has no normals/shading):**
  1. *Cheap first cut:* Retinex-style normalization — divide out heavily-blurred L channel in UV
     space on both textures before differencing; do the difference in **chromaticity** (Lab a/b or
     log-RGB ratios), which is far more lighting-invariant than raw color.
  2. *Proper fix:* fit the **makeup-priors** model (PCA prior first) instead of raw per-pixel
     difference. See references.md.
- **Data: use FFHQ-Makeup** (synthetic same-identity with/without pairs, multi-style) — unblocks
  the "needs real same-person pairs" dependency without self-capture.

**Phase 2 — Application pipeline (`apply.py`).**
- warp + skin-tone-aware blend onto target; add/fine-tune the **harmonization refiner**.
  Verify across skin tones.
- **Per-region blend modes** (stored in look `meta.json`): lipstick ≈ absorptive (multiply-like),
  powder shadow ≈ scattering (soft-light-like), glitter ≈ additive-specular. One global Lab blend
  is physically wrong; parsing masks make per-region params cheap.
- **Refiner:** prefer a small U-Net trained on **synthetic composites** (apply a layer with
  randomized params → learn to fix seams/graininess), or optional low-strength SD img2img
  "enhance" pass. Avoid full diffusion transfer (SHMT/Stable-Makeup) as a pipeline component —
  end-to-end, kills the portable layer; use them as **quality baselines** only.

**Phase 2.5 — Single-image extraction (promoted from "deferred").**
- Biggest UX barrier is the pair requirement — consumers want "steal this look from any photo."
- Makeup removal (SSAT or diffusion editor) synthesizes the bare face → feed the **existing**
  pair pipeline as a pseudo-pair. Zero architecture changes.

**Phase 3 — Local save/load + demo.**
- `storage.py` (local-folder fake) + Gradio app: upload pair → extract → save look → upload
  target → apply → view/download result; list & re-apply saved looks.
- **Per-region intensity sliders** (lips/eyes/skin = per-region alpha scaling — nearly free with
  the layer representation, top-loved feature in commercial try-on apps).
- **Instant preview, async polish:** classical warp+blend renders sub-second; run the refiner as
  a background "enhance". **Fail loudly:** gate on pose/landmark quality at upload with
  actionable feedback ("face the camera, even lighting") instead of producing a smeared result.
- **Look previews:** render each saved look onto a neutral bare-face thumbnail for the library.

## Risks & mitigations
- **Alpha picks up shadow/lighting** (#1 risk: our 2D UV unwrap has no normals → no shading
  estimate) → chromaticity difference + Retinex normalization first; makeup-priors fitting as the
  proper fix; UV symmetry; clip to parsed regions. If UV quality itself is the bottleneck →
  **FreeUV** (CVPR 2025) or DECA.
- **Thin/high-freq makeup smears** (winged eyeliner, glitter — 468-pt mesh is coarse at eyes)
  → higher UV res in eye region; CPM-style separate pattern branch; mirror-fill only fills holes,
  never averages into observed pixels.
- **Result looks pasted-on** → fine-tune refiner; physically-based translucent blend.
- **Non-frontal/occluded faces** → UV helps; gate to reasonably frontal photos in MVP.
- **GPU latency/cost** → run offline/batch locally for now; no realtime requirement yet.

## Verification (local, end-to-end)
1. **Phase 0 gate:** hand-built UV layer re-applies plausibly to a third face, skin tone shows through.
2. **Extraction:** RGBA layer covers correct regions with clean alpha on held-out pairs.
3. **Application:** apply one look to faces of **different skin tones** → adapts, identity preserved.
4. **Full local loop:** Gradio demo — extract from a pair → save look → apply saved look to a new
   no-makeup photo → download result. All on local files only.

## Progress log

- **2026-06-03 — Scaffold + preprocessing done (Phase 0 partial).**
  - Project scaffold, `requirements.txt`, `.gitignore`, `src/makeup_transfer/` package.
  - `storage.py`: local LookStore (RGBA layer + `meta.json`, zip-to-share), swappable interface.
  - `preprocess.py`: **working** — MediaPipe **Tasks API** `FaceLandmarker` (legacy `mp.solutions`
    is gone in 0.10.35); model at `weights/face_landmarker.task`. 468 landmarks + region masks
    (lips/eyes/brows/face). Verified on `data/samples/face_*.jpg`.
  - venv `.venv` with core deps (numpy<2, opencv, pillow, scipy, scikit-image, mediapipe). **torch
    CUDA not yet installed.** Env: Python 3.12.10, RTX 4060 Ti (8GB).
  - **NEXT:** 3D fit → UV unwrap (3DDFA_V2/DECA) — riskiest dep; decide native Win vs Conda/WSL vs
    Colab. Then Phase 0 gate (hand-derive one layer, re-apply), then `extract.py` / `apply.py`.

- **2026-06-12 — Phase 0 gate PASSED (MediaPipe canonical-UV route).**
  - torch 2.6.0+cu124 installed and verified on the 4060 Ti (`torch.cuda.is_available() == True`).
  - 3DDFA_V2 cloned to `third_party/` (weights + BFM bundled in repo), but **building its native
    extensions was blocked** by the sandbox permission classifier (compiling external repo code
    needs explicit authorization). Workaround that turned out fine on its own:
  - `uv_unwrap.py` — **working** UV unwrap/re-projection with NO third-party code: MediaPipe 468
    landmarks + canonical face model UVs (`weights/canonical_face_model.obj`), per-triangle
    barycentric warp in numpy/cv2. Includes per-triangle visibility (occlusion) + left/right
    mirror fill for turned faces. 2D piecewise-affine, not a true 3DMM — revisit 3DDFA_V2/DECA
    if UV quality becomes the bottleneck.
  - `phase0_spike.py` — hand-derived RGBA layer from an MT-dataset makeup face (color = UV tex,
    alpha = Lab distance to inpainted bare-skin estimate, gated to lip/eye regions), saved via
    `LookStore` (`looks/spike-red-gold/`), applied to a different person with a skin-tone-aware
    Lab blend (target keeps own luminance base + makeup's relative darkening; chroma alpha-mixed).
  - **Gate verdict: layer concept is plausible.** Red lipstick transfers convincingly onto much
    paler skin — lip texture/highlights show through, identity preserved. Weak spots: eyeshadow
    alpha too faint (source was 3/4 pose; gold shadow diluted by mirror fill), eye-band gating
    is crude. Outputs: `preprocess_out/phase0/`.
  - Test data note: `data/samples/face_*.jpg` are StyleGAN faces of **children** — fine for
    landmark tests, not used for makeup work. Adult MT-dataset samples in `data/mt/`.
  - **NEXT (Phase 1, `extract.py`):** pair-based extraction in UV space — illumination-aware
    difference of with/without textures instead of the hand gate; needs real same-person pairs
    (makeup-extract repo data or self-captured). Reuse `uv_unwrap` as-is.

- **2026-07-02 — Research review; plan updated (no code changes).**
  - Verdict: architecture (RGBA layer in UV space, pair-first, reuse pretrained) confirmed sound —
    aligned with where the makeup-extract group went next. Key updates folded into this plan +
    references.md:
  - **Phase 1 blueprint changed** to **makeup-priors** (CVPR 2024, same authors as makeup-extract):
    fit a learned makeup prior in UV space instead of raw differencing. Cheap first cut before
    that: chromaticity difference + Retinex (blurred-L) normalization.
  - **Pair-data dependency unblocked:** **FFHQ-Makeup** (2025) — synthetic same-identity
    with/without pairs, multi-style.
  - **New refs added:** FreeUV (CVPR 2025, UV recovery upgrade path), SHMT (NeurIPS 2024) +
    FLUX-Makeup/DreamMakeup/BeautyDiffusion as diffusion baselines (end-to-end → baselines only,
    not pipeline components).
  - **Roadmap re-ranked by UX leverage:** single-image extraction promoted to Phase 2.5
    (removal → pseudo-pair, zero architecture changes); Phase 3 gains per-region intensity
    sliders, instant-preview + async refine, pose gating with actionable feedback, look
    thumbnails. Phase 2 gains per-region blend modes in `meta.json`.
  - **NEXT (unchanged): Phase 1 `extract.py`** — start with chromaticity+Retinex difference on an
    FFHQ-Makeup pair, then evaluate makeup-priors PCA fitting.

- **2026-07-02 — Phase 1 extraction pipeline working (`extract.py`).**
  - **Data unblocked:** FFHQ-Makeup samples downloaded from the `images/` folder of
    HF `cyberagent/FFHQ-Makeup` (no need for the full 3.8 GB zip) → `data/ffhq_makeup/`,
    3 identities × (bare + 5 styles), CC BY-NC-SA. See `data/ffhq_makeup/README.md`.
  - `extract.py` — pair-based extraction as planned (cheap first cut, no priors yet):
    both photos → UV (mirror-filled) → **chromaticity difference** (Lab a/b, global median
    drift removed) + **Retinex-normalized L** darkening/brightening terms → smoothstep alpha,
    gated to face minus eye/mouth openings, visibility-weighted, **UV-symmetry** min-mix
    (`--sym`, default 0.5) → RGBA layer via `LookStore`. Writes debug maps
    (`preprocess_out/phase1/`); `--apply-to` re-applies for verification (phase0 blend).
  - `preprocess.py`: added `LIPS_INNER` / `"mouth"` region (mouth-interior exclusion).
  - **Bug found+fixed:** final alpha blur bled lip alpha into the thin mouth-interior UV
    sliver; open-mouthed targets stretch that sliver across the teeth (plum teeth). Hard
    exclusions (mouth/eyes) are now re-asserted after all blurs.
  - **Verified on 3 held-out pairs** (002386-03, 003125-02, 003417-05), cross-applied between
    identities: lipstick, brow fill and smoky eye transfer convincingly; teeth/eyes clean;
    identity preserved. Phase-1 gate (clean alpha, correct regions) met.
  - Known limitations: source wearing **glasses** leaves symmetric rim ghosts in the matte
    (symmetry suppression can't remove them → needs input gating, already planned Phase 3);
    foundation-over-stubble reads as chin darkening; eyeshadow transfers subtler than source.
  - **NEXT:** Phase 2 `apply.py` (per-region blend modes, refiner) — or first evaluate
    makeup-priors PCA fitting as the extraction upgrade.

## Open questions to settle during build
- Refiner: small U-Net on synthetic composites vs fine-tune CPM's decoder vs low-strength SD
  img2img "enhance".
- makeup-priors: PCA prior (fast, low-freq) vs StyleGAN2 prior (high-freq, heavier) — start PCA.
- UV upgrade path if quality bottlenecks: FreeUV vs DECA vs 3DDFA_V2 (native build was
  sandbox-blocked; see 2026-06-12 entry).
- Single-image extraction (Phase 2.5): SSAT removal vs diffusion-editor removal for the bare-face
  estimate.
