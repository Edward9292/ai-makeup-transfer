# Design: Phase 2.5 — Single-Image Extraction (removal → pseudo-pair)

**Date:** 2026-07-02
**Status:** approved (spike-first)
**Depends on:** Phase 1 `extract.py` (pair extraction), Phase 2 `apply.py` (verification blend)

## Problem

The extraction pipeline requires a with-makeup **and** a without-makeup photo of the same
person. That pair requirement is the biggest UX barrier — consumers want "steal this look
from any photo." Phase 2.5 removes it by synthesizing the bare face from the with-makeup
photo alone, then feeding the existing pair pipeline as a **pseudo-pair**. Zero changes to
the extraction math.

## Decisions made

- **Bare-face estimator: SSAT makeup removal** (AAAI 2022, `github.com/Snowfallingplum/SSAT`),
  chosen over a diffusion instruct-editor. Rationale: in `extract.py` the layer's *color*
  comes from the with-makeup photo at full resolution — the bare estimate only drives the
  **alpha matte** and the skin-mean reference. It therefore needs faithful low-frequency skin
  color and rock-solid identity/geometry, not fine detail. SSAT is purpose-built for removal,
  identity-preserving, deterministic, pure-PyTorch inference (no native builds — avoids the
  3DDFA sandbox block), and trivially fits the 8 GB RTX 4060 Ti. Its 256×256 output mostly
  costs alpha detail, which the Retinex/chromaticity steps blur anyway.
- **Diffusion editor is the documented fallback** if SSAT fails the gate — not built now.
  Identity/color drift (which differencing amplifies into false alpha) makes it second choice.
- **Spike first, integrate after the gate** (same pattern as Phase 0): prove removal quality
  on FFHQ-Makeup samples with ground-truth bare faces before touching `extract.py`.

## Architecture / data flow

```
with-makeup photo ─► removal.estimate_bare() ─► pseudo-bare photo ┐
        │                                                          ├─► existing extract.py
        └──────────────────────────────────────────────────────────┘    (UV chromaticity diff)
```

Because the pseudo-bare is the *same photograph* with makeup removed, both images share
identical geometry and landmarks — image-space alignment is better than a real pair.

SSAT detail: its removal branch is a style transfer that needs a bare **reference** face as
the "no makeup" style source. We use one fixed neutral FFHQ bare image as that reference.

## Components

1. **`third_party/SSAT`** — cloned repo + pretrained weights. Inference only; we import its
   network definitions, never compile anything. Spike verifies: weights availability, torch
   2.6 compatibility (may need `weights_only=False` shims), license terms, input alignment
   requirements.
2. **`src/makeup_transfer/removal.py`** — `estimate_bare(image_bgr) -> image_bgr`. Crops and
   aligns the face to SSAT's expected input, runs removal at 256×256, upsamples and pastes
   the result back into the full frame. SSAT stays isolated behind this interface so the
   fallback (diffusion editor) can replace it without touching callers.
3. **`src/makeup_transfer/phase25_spike.py`** — for 2–3 FFHQ-Makeup identities:
   extract a layer from (makeup, pseudo-bare) and from (makeup, true-bare); write
   side-by-side alpha mattes, per-pixel diff maps, and a cross-apply grid (both layers
   applied to a different identity via `apply.py`) to `preprocess_out/phase25/`.
4. **After the gate passes — `extract.py --single`:** the bare-photo argument becomes
   optional; when absent, call `removal.estimate_bare`. `meta.json` records
   `source_mode: "pair" | "single"`. No other changes.

## Gate (go/no-go, mirrors Phase 0)

The pseudo-pair layer, judged against the ground-truth-pair layer on held-out FFHQ identities:

1. Alpha covers the **same makeup regions** (lips/brows/eyes) — no major missed makeup.
2. **No large false alpha** on bare skin (the failure mode of skin-tone drift in the removal).
3. Cross-applied to a different identity, the result is **plausible and comparable** to the
   pair-based application (eyeball verdict, grid image as evidence).

If the gate fails: document the failure mode, evaluate the diffusion-editor fallback as a
separate decision — do not integrate a broken removal.

## Error handling

- Missing SSAT weights → fail loudly with download instructions (no silent degradation).
- Face detection failure on input or pseudo-bare → existing loud error path in `preprocess.py`.
- No automatic removal-quality detection in this phase; known failure modes get documented in
  the progress log (input gating is already planned for Phase 3).

## Verification

- Reuse the 3 held-out FFHQ-Makeup pairs from Phase 1 (002386-03, 003125-02, 003417-05) —
  ground-truth bare faces make the layer comparison direct.
- Deliverables: `preprocess_out/phase25/` comparison grids + a progress-log entry in
  `BUILD_PLAN.md` with the gate verdict.

## Out of scope

- Diffusion-editor removal (fallback only, separate decision if needed).
- Removal-quality auto-gating at upload (Phase 3).
- makeup-priors PCA fitting (separate extraction upgrade).
- Retraining/fine-tuning SSAT.
