# Design: PCA makeup prior for extraction (Phase 1 upgrade)

**Date:** 2026-07-14
**Status:** approved, ready for implementation planning
**Supersedes:** the "cheap first cut" extraction in `extract.py` (chromaticity difference +
Retinex normalization), which stays in the tree as the baseline to compare against.

## Problem

`extract.py` builds the makeup alpha by *differencing* a with-makeup UV texture against a bare UV
texture. Anything that differs between the two shots reads as makeup. Two failures follow from
that, both already documented in `BUILD_PLAN.md`:

1. **Phase 2.5 / 003417 skin-tone drift.** In single-photo mode the "bare" side is synthesized by
   SSAT, which shifts the subject's skin tone (domain shift — SSAT was trained on the mostly-female
   MT dataset). The difference then reads makeup across the *entire* face: false-alpha 0.17,
   roughly 8x the other two held-out identities, and the applied result carries a full-face color
   cast.
2. **Shadows, exposure and glasses ghosts** contaminate the alpha matte generally. Retinex
   normalization, a global median a/b shift, and UV-symmetry min-mixing are hand-tuned patches
   over this, not a model of it.

BUILD_PLAN has named the makeup-priors model (CVPR 2024, same authors as makeup-extract) as the
proper fix in three consecutive progress entries. This spec designs it.

## Core idea

Learn what makeup *can* look like, then refuse to explain anything else as makeup. A makeup layer
is constrained to lie (mostly) in a learned low-dimensional subspace; shadows and skin-tone drift
do not live in that subspace, so a fit to it cannot represent them and they fall out.

**The wrinkle that shapes the whole design:** foundation is full-face makeup, and SSAT's skin-tone
drift is *also* a full-face color shift. The PCA basis will learn a "whole face shifts warmer"
component, because foundation is in the training data — and that component would happily absorb the
drift and call it foundation. **A naive prior fit does not fix 003417.** The separation has to be
explicit, and the signal available is smoothness: shadow and skin-drift are smooth and low-order
across the face; makeup has structure (it stops at the jaw, it concentrates on lips/eyes/brows).
Hence the nuisance basis below, which is the load-bearing part of this design.

## Architecture

One new module, `src/makeup_transfer/priors.py`. `extract.py` gains an alternative alpha path but
is not restructured — it is already 281 lines and its docstring committed to keeping this swap
local. **Unchanged: `apply.py`, `storage.py`, `uv_unwrap.py`, `preprocess.py`, the look format,
and the Gradio app.** The prior changes *how alpha is computed*, not what a look is, so every
saved look and the entire apply side keep working.

### Components

**`MakeupPrior`** — the learned model. Holds `mean`, `components` (K x D), per-component
`variances`, the UV resolution and face mask it was built against. `save()` / `load()` to
`weights/makeup_prior_pca.npz`. `weights/` is gitignored (repo convention: every weight is a build
artifact), so the model is produced by a script, not committed.

**`build_prior(pairs, ...)`** — corpus builder. For each training pair: unwrap both photos to UV
via the existing `unwrap_to_uv`, take the **raw Lab difference** (see below), mask to the face
region, flatten. Stack, then PCA (randomized SVD). K chosen by explained variance (target 95%,
hard cap so it stays well below the sample count).

**`fit(d_obs, ...)`** — the robust fit. Returns the reconstructed makeup field and the fitted
coefficients (both makeup and nuisance, for debugging).

**`extract_layer_prior(...)`** — drop-in alternative to `extract_layer` in `extract.py`, same
signature shape and same return contract `(layer_rgba, skin_lab_mean, debug)`.

**`priors_eval.py`** — the gate harness, mirroring `phase25_spike.py`.

## The difference field

**Raw Lab difference — no Retinex, no median shift.** Those two exist only to paper over
illumination drift, and the nuisance basis is the principled version of them. Stacking them would
mean correcting the same nuisance twice, in a way the prior can no longer see or undo. So the
prior fit *replaces* them.

Difference maps are built and fitted at **128x128** (the prior is inherently low-frequency; this
keeps the PCA in RAM at roughly 300 MB for ~2,500 samples) and masked to the face region. The
reconstructed field is upsampled to the 512 working resolution afterwards.

## The fit (the core)

Given an observed UV difference `d_obs`, solve one robust least-squares problem:

```
d_obs  ≈  mean + M·c  +  N·k
             └ makeup ┘   └ nuisance ┘
```

- `M` — the PCA makeup basis.
- `N` — the nuisance basis: a per-channel constant plus a degree-2 polynomial in UV coordinates.
  This is the explicit model of "smooth shading, exposure, white balance, and skin-tone drift."
- `c` is shrunk toward zero under the PCA variances (ridge weighted by inverse eigenvalue). This
  shrinkage is what makes it a *prior* rather than a bare projection: implausible makeup is
  penalized in proportion to how implausible it is.
- Solved by IRLS with Huber weights, so a few bad pixels cannot drag the fit.

**Then keep `mean + M·c` and discard `N·k`.** That discard is the 003417 fix.

### The residual term

A linear low-rank basis cannot represent a winged eyeliner edge or glitter speckle. The
unexplained remainder `r = d_obs − (mean + M·c + N·k)` holds exactly that detail — but also noise,
and any *sharp* nuisance the degree-2 polynomial could not absorb. So it is not added back raw.
Two filters:

- **High-pass:** `r_hi = r − blur(r)`. The nuisance basis already took the smooth drift; this is a
  second guard against slow drift leaking back in through the residual.
- **Support gate:** the reconstructed field `mean + M·c` says where the prior believes makeup
  actually is. `g` is a softened support mask derived from it, and detail is added only there.

```
makeup_final = mean + M·c + w · g · r_hi
```

The prior therefore does two jobs: it rejects smooth nuisance, and it **decides where detail is
allowed to exist**. Raw pixels only supply detail inside territory the prior has vouched for.

**`w = 0` reproduces the prior-only model exactly.** This is deliberate. We implement `w = 0`
first, and the eval *sweeps* `w ∈ {0, 0.25, 0.5, 1.0}` against the gate metrics — so we measure
what the residual buys (sharper detail) and what it costs (false-alpha) rather than assuming.
Prior-only remains a live configuration, not a deleted branch.

## A bonus: drift-free `skin_lab_mean`

Once the makeup field is reconstructed, the estimated bare skin is
`observed_makeup_Lab − makeup_field` — derived from the **real photo**, not the pseudo-bare. So
`skin_lab_mean` (which `apply.py` uses as the reference every blend is relative to) no longer
depends on the pseudo-bare's absolute skin tone at all. SSAT's drift stops leaking into the
apply-time blend as well as into the alpha.

## Data flow

**Train (one-time):**
```
FFHQ-Makeup pairs → unwrap both to UV → raw Lab diff (128²), face-masked
                  → stack → PCA → weights/makeup_prior_pca.npz
```

**Extract:**
```
uv_with, uv_without (real or SSAT pseudo-bare)
  → raw Lab diff → robust fit [M | N] → drop N·k
  → + w·g·r_hi → upsample to 512 → smoothstep alpha
  → EXISTING gates unchanged (mouth/eye hard exclusion, visibility weighting)
  → RGBA layer;  skin_lab_mean from the bare estimate above
```

UV-symmetry suppression becomes **off by default** (shadow rejection is now the prior's job) but
stays available as a flag — see the glasses risk below.

## Training data

**~300–500 FFHQ-Makeup identities** (~1,500–2,500 pairs), fetched as individual files from the
HuggingFace repo (`cyberagent/FFHQ-Makeup`, `images/`) rather than the full 3.8 GB zip — the same
route already used for the 3 sample identities. CC BY-NC-SA 4.0, non-commercial.

The 3 identities currently in `data/ffhq_makeup/` (002386, 003125, 003417) are the **held-out eval
set and must be excluded from the training corpus.** `data/ffhq_makeup/*` is gitignored apart from
its README, so the bulk pull does not enter git.

## Rollout

`extract.py --prior` is **opt-in**; the current extraction stays the default. If the gate passes,
the default flips and `BUILD_PLAN.md` records it. Both paths stay in the tree so the eval can keep
comparing them. New flags: `--prior`, `--prior-model <path>`, `--prior-lambda`, `--residual <w>`.

## Error handling

- **Missing prior model** → `FileNotFoundError` naming the build command; in the Gradio app, a
  `gr.Error`, matching how the existing GPU-missing case is handled.
- **No silent fallback.** If the prior was requested and cannot run, it says so. Falling back to
  the baseline extractor silently would make the eval meaningless.
- **Corpus build** skips pairs where face detection fails and *reports the count* rather than
  silently dropping them.
- **Degenerate fit** (K >= sample count, singular design matrix) is caught at build/load time with
  an explicit error, not at fit time.

## Testing

The load-bearing test is **nuisance rejection** — it tests the 003417 mechanism directly and needs
no real data:

- **Nuisance rejection:** synthesize a UV pair with *zero* makeup but a global color offset plus a
  smooth gradient applied to the bare side. Assert the recovered makeup field is ≈ 0. The baseline
  chromaticity difference fails this by construction; that contrast is the point. Must still pass
  at the chosen `w` (with smooth-only nuisance, `r_hi ≈ 0`).
- **High-frequency recovery:** synthesize a pair with a thin stroke the basis provably cannot
  represent. Assert it is missed at `w = 0` and recovered at `w > 0`.
- **Basis round-trip:** a difference synthesized from a known basis component is recovered by the
  fit.
- **`MakeupPrior` save/load** round-trip.
- **Integration:** `extract_look(..., prior=...)` on a real FFHQ pair produces a layer with alpha
  concentrated in the lip region. Uses a *tiny* fixture prior (few components, built from a handful
  of pairs) so the suite stays fast.

## The gate

`priors_eval.py`, mirroring `phase25_spike.py`: baseline vs prior, on the 3 held-out identities, in
both pair and single mode, with the same metrics (mean |dA| / false-alpha / missed over the UV face
region) plus a comparison grid.

**Pass = 003417's false-alpha (currently 0.17) drops to roughly the level of the other two
(~0.02–0.04), with no regression on 002386 (0.0221) or 003125 (0.0389), and the applied results
hold up to a human eyeball.** Only then does the prior become the default.

## Known limitations and risks

- **Glasses/sharp-edge risk from the residual (the main new risk).** Glasses rims and sharp cast
  shadow edges are *high-frequency*, so the high-pass will not stop them — the support gate is what
  has to. Rims sit directly on the eye band, exactly where an eyeshadow look's support is high, so
  the residual could reintroduce the known glasses-ghosting limitation that prior-only would have
  suppressed. **003417 wears glasses, so the eval will expose this if it is real — its false-alpha
  is the canary.** If the residual reopens it: keep symmetry suppression on for the residual term
  only (rims are symmetric and so is most makeup, but a min-mix still suppresses a rim harder than
  a wing), or cap `w` within the eye band.
- **Foundation vs. drift is genuinely ambiguous.** The nuisance basis separates them by
  smoothness, which is a real but imperfect signal. A perfectly uniform, jaw-to-jaw foundation with
  no edge is, to this model, indistinguishable from a skin-tone shift — and will be partly
  discarded. This is an accepted trade: losing some foundation strength is much cheaper than a
  full-face color cast.
- **A linear basis is low-rank.** Even with the residual, expect makeup detail to be softer than
  the raw difference produces. StyleGAN2-prior (the paper's expressive option) is the documented
  escalation, out of scope here.
- **Synthetic training data.** The prior is learned from FFHQ-Makeup, which is *synthetic* makeup.
  Real-world makeup outside that generator's distribution will fit less well, and the failure is
  quiet — under-fitting looks like a washed-out layer, not an error.

## Out of scope

StyleGAN2 makeup prior; the harmonization refiner (deferred 2026-07-02, still deferred); UV
upgrade (FreeUV/DECA); darker-skin verification data pull; upload-time pose gating.
