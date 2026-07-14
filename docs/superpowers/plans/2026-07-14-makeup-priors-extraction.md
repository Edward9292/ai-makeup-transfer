# PCA Makeup Prior Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the raw chromaticity-difference makeup alpha with a fitted PCA makeup prior in UV space, so that shadows and SSAT's skin-tone drift can no longer be mistaken for makeup.

**Architecture:** One new module `src/makeup_transfer/priors.py` holds the learned model (`MakeupPrior`), the robust fit, and the corpus builder. `extract.py` gains an opt-in `--prior` alpha path; nothing else in the pipeline changes. The fit explains an observed UV difference as `mean + M·c + N·k` — a learned makeup basis plus an explicit smooth-nuisance basis — then **discards `N·k`**. That discard is the whole point: it is what removes the skin-tone drift. A gated high-frequency residual (`w` knob, `w=0` = prior-only) adds back detail a linear basis cannot represent.

**Tech Stack:** Python 3.12, numpy (<2.0), OpenCV, scikit-image, pytest. **No new dependencies** — PCA is done with the Gram-matrix trick in numpy (samples ≪ dimensions), so scikit-learn is not needed.

**Spec:** `docs/superpowers/specs/2026-07-14-makeup-priors-extraction-design.md`

## Global Constraints

- **Do not modify** `apply.py`, `storage.py`, `uv_unwrap.py`, `preprocess.py`, or the look format. The prior changes *how alpha is computed*, not what a look is. Existing saved looks and the Gradio app must keep working untouched.
- **UV working resolution is 512** (`extract.UV_SIZE`); the **prior is built and fitted at 128** and its reconstruction upsampled to 512.
- **The difference field is a RAW Lab difference** — no Retinex, no median a/b shift. The nuisance basis replaces both. Do not stack the old corrections under the new fit.
- **Held-out identities — never train on these:** `002386`, `003125`, `003417`. They are the eval set.
- **No new pip dependencies.** Use numpy/scipy/OpenCV/scikit-image only.
- **No silent fallback:** if the prior is requested and cannot run, raise. Falling back to the baseline extractor would make the eval meaningless.
- `weights/` and `data/ffhq_makeup/*` are **gitignored**. The prior model is a build artifact; never `git add` it or the corpus.
- Region masks come from `uv_unwrap.region_masks_uv(size)`, which returns uint8 0/255 masks keyed `face`, `left_eye`, `right_eye`, `mouth`, `lips`, `brows` (see `preprocess.REGION_GROUPS`).

## File Structure

| File | Responsibility |
|---|---|
| `src/makeup_transfer/priors.py` | **New.** `MakeupPrior` (mean/components/variances, save/load), nuisance basis construction, robust IRLS fit, gated residual, `build_prior()` corpus builder, `extract_layer_prior()`, CLI (`--build`). |
| `src/makeup_transfer/fetch_ffhq.py` | **New.** Extracts a training subset of identities from `FFHQ-Makeup.zip`, excluding the held-out three. |
| `src/makeup_transfer/extract.py` | **Modify.** Add `prior=`/`residual=` params to `extract_look`/`extract_look_single`, route alpha through `extract_layer_prior` when a prior is given, add CLI flags. |
| `src/makeup_transfer/priors_eval.py` | **New.** The gate harness: baseline vs prior on the 3 held-out identities, pair + single mode, sweeping `w`. |
| `tests/test_priors.py` | **New.** Nuisance rejection, high-frequency recovery, basis round-trip, save/load. All synthetic — no data, no GPU. |
| `tests/test_extract_prior.py` | **New.** Integration: `extract_look(..., prior=...)` on a real FFHQ pair with a tiny fixture prior. |

---

### Task 1: Extract a training corpus from the zip

The download is **already complete**: `data/FFHQ-Makeup.zip`, 3.6 GB, 128,590 members, verified 2026-07-14.

**Files:**
- Create: `src/makeup_transfer/fetch_ffhq.py`

**Interfaces:**
- Produces: `HELDOUT: set[str]`, `extract_subset(zip_path, out_dir, n_identities) -> list[str]` (returns the identity IDs written).

**Verified zip layout — do not assume otherwise.** Members are **one directory per identity**, not flat filenames:

```
000000/
000000/bare.jpg
000000/makeup_01.jpg
000000/makeup_02.jpg
```

The rest of the pipeline (`find_pairs`, and the existing `data/ffhq_makeup/`) uses the **flat** `<id>_bare.jpg` / `<id>_makeup_NN.jpg` convention. So the extractor must **flatten and rename** as it writes.

- [ ] **Step 1: Write the extractor**

```python
"""Extract a training subset of FFHQ-Makeup identities from the full zip.

The HF `images/` folder holds only 20 identities; the full 18k live only inside
FFHQ-Makeup.zip. We pull a subset out of the zip and leave the rest alone.

The zip stores one directory per identity (`000000/bare.jpg`, `000000/makeup_01.jpg`), but the
rest of this project uses flat `<id>_bare.jpg` / `<id>_makeup_NN.jpg` names — as does the existing
`data/ffhq_makeup/` sample dir, which `priors.find_pairs` globs. So we flatten on the way out.

Run::

    PYTHONPATH=src python -m makeup_transfer.fetch_ffhq --n 400
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

ZIP = Path("data/FFHQ-Makeup.zip")
OUT = Path("data/ffhq_makeup_train")

# Eval identities. Training on these would invalidate the gate.
HELDOUT = {"002386", "003125", "003417"}

# e.g. "000000/makeup_01.jpg" -> ("000000", "makeup_01")
_MEMBER_RE = re.compile(r"(?:^|/)(\d{6})/(bare|makeup_\d+)\.jpg$")


def extract_subset(zip_path: Path = ZIP, out_dir: Path = OUT, n_identities: int = 400) -> list[str]:
    """Write `n_identities` identities' photos into `out_dir`, flattened. Returns the IDs written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members: dict[str, list[tuple[str, str]]] = {}   # ident -> [(member_path, kind)]
        for name in z.namelist():
            m = _MEMBER_RE.search(name)
            if not m:
                continue
            ident, kind = m.group(1), m.group(2)
            if ident in HELDOUT:
                continue
            members.setdefault(ident, []).append((name, kind))

        chosen = sorted(members)[:n_identities]
        for ident in chosen:
            for member, kind in members[ident]:
                # 000000/makeup_01.jpg -> 000000_makeup_01.jpg
                (out_dir / f"{ident}_{kind}.jpg").write_bytes(z.read(member))
    return chosen


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Extract an FFHQ-Makeup training subset from the zip.")
    ap.add_argument("--zip", default=str(ZIP))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n", type=int, default=400, help="number of identities")
    args = ap.parse_args()

    ids = extract_subset(Path(args.zip), Path(args.out), args.n)
    n_files = len(list(Path(args.out).glob("*.jpg")))
    print(f"Extracted {len(ids)} identities ({n_files} photos) to {args.out}")
    print(f"Held out (never trained on): {sorted(HELDOUT)}")


if __name__ == "__main__":
    _main()
```

- [ ] **Step 2: Gitignore the training corpus**

Add to `.gitignore`:

```
data/ffhq_makeup_train/
```

- [ ] **Step 3: Run it and verify the held-out identities are absent**

```bash
PYTHONPATH=src python -m makeup_transfer.fetch_ffhq --n 400
ls data/ffhq_makeup_train | wc -l          # expect ~2400 files (400 x ~6)
ls data/ffhq_makeup_train | grep -E "002386|003125|003417" | wc -l   # MUST print 0
```

Expected: ~2400 files, and **0** held-out files. If the last command prints anything but 0, the eval is compromised — stop and fix.

- [ ] **Step 4: Commit**

```bash
git add src/makeup_transfer/fetch_ffhq.py .gitignore
git commit -m "Phase 4: FFHQ-Makeup training subset extractor (held-out identities excluded)"
```

---

### Task 2: `MakeupPrior` — the model container

**Files:**
- Create: `src/makeup_transfer/priors.py`
- Test: `tests/test_priors.py`

**Interfaces:**
- Produces:
  - `PRIOR_SIZE = 128`, `PRIOR_PATH = Path("weights/makeup_prior_pca.npz")`
  - `face_mask_flat(size) -> np.ndarray` — bool (size*size,), the UV face region.
  - `class MakeupPrior` with fields `mean` (D,), `components` (K, D), `variances` (K,), `size` (int); methods `save(path)`, `load(path)` (classmethod), property `n_components`.
  - `D == 3 * face_mask_flat(size).sum()` — the flattened, face-masked Lab difference (channels L, a, b concatenated).

- [ ] **Step 1: Write the failing tests**

`tests/test_priors.py`:

```python
"""Prior model: container, fit, nuisance rejection, residual. All synthetic — no data, no GPU."""
import numpy as np
import pytest

from makeup_transfer.priors import MakeupPrior, PRIOR_SIZE, face_mask_flat


def test_face_mask_flat_shape():
    m = face_mask_flat(PRIOR_SIZE)
    assert m.shape == (PRIOR_SIZE * PRIOR_SIZE,)
    assert m.dtype == bool
    assert 0 < m.sum() < m.size          # a real subset of the chart


def test_prior_save_load_roundtrip(tmp_path):
    d = 3 * int(face_mask_flat(PRIOR_SIZE).sum())
    rng = np.random.default_rng(0)
    prior = MakeupPrior(
        mean=rng.normal(size=d).astype(np.float32),
        components=rng.normal(size=(4, d)).astype(np.float32),
        variances=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        size=PRIOR_SIZE,
    )
    path = tmp_path / "prior.npz"
    prior.save(path)
    back = MakeupPrior.load(path)

    assert back.size == PRIOR_SIZE
    assert back.n_components == 4
    np.testing.assert_allclose(back.mean, prior.mean)
    np.testing.assert_allclose(back.components, prior.components)
    np.testing.assert_allclose(back.variances, prior.variances)


def test_load_missing_file_names_the_build_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="--build"):
        MakeupPrior.load(tmp_path / "nope.npz")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_priors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'makeup_transfer.priors'`

- [ ] **Step 3: Write the module skeleton**

`src/makeup_transfer/priors.py`:

```python
"""Phase 4: a learned PCA makeup prior in UV space.

The Phase-1 extractor *differences* a makeup UV texture against a bare one, so anything that
differs between the two shots reads as makeup — shadows, exposure, and (in single-photo mode)
SSAT's skin-tone drift. This module replaces that with a fit:

    d_obs  ~=  mean + M.c  +  N.k
                  \\makeup/    \\nuisance/

`M` is a PCA basis learned from real makeup differences; `N` is an explicit smooth-nuisance basis
(per-channel constant + degree-2 polynomial in UV coords) modelling shading, exposure, white
balance and skin-tone drift. We keep `mean + M.c` and **throw `N.k` away** — that discard is the
fix for the Phase-2.5 003417 failure.

Foundation is full-face makeup and skin drift is a full-face colour shift, so the two are
genuinely ambiguous; the nuisance basis separates them by *smoothness* only. A perfectly uniform,
edgeless foundation is partly discarded along with the drift. That is the accepted trade — losing
some foundation strength is much cheaper than a full-face colour cast.

Run::

    PYTHONPATH=src python -m makeup_transfer.priors --build --data data/ffhq_makeup_train
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PRIOR_SIZE = 128           # the prior is inherently low-frequency; 128 keeps the PCA in RAM
PRIOR_PATH = Path("weights/makeup_prior_pca.npz")

_BUILD_HINT = (
    "No makeup prior at {path}. Build one first:\n"
    "    PYTHONPATH=src python -m makeup_transfer.priors --build --data data/ffhq_makeup_train"
)


def face_mask_flat(size: int = PRIOR_SIZE) -> np.ndarray:
    """The UV face region as a flat boolean mask of length size*size."""
    from .uv_unwrap import region_masks_uv

    return (region_masks_uv(size)["face"] > 0).reshape(-1)


@dataclass
class MakeupPrior:
    """A PCA basis over face-masked Lab makeup-difference fields."""

    mean: np.ndarray          # (D,)
    components: np.ndarray    # (K, D)
    variances: np.ndarray     # (K,) per-component variance, descending
    size: int                 # UV resolution the basis was built at

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def save(self, path: str | Path = PRIOR_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, mean=self.mean, components=self.components,
            variances=self.variances, size=np.int32(self.size),
        )

    @classmethod
    def load(cls, path: str | Path = PRIOR_PATH) -> "MakeupPrior":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(_BUILD_HINT.format(path=path))
        z = np.load(path)
        return cls(
            mean=z["mean"], components=z["components"],
            variances=z["variances"], size=int(z["size"]),
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_priors.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/makeup_transfer/priors.py tests/test_priors.py
git commit -m "Phase 4: MakeupPrior model container (save/load, face mask)"
```

---

### Task 3: The nuisance basis and the robust fit

This is the load-bearing task. The nuisance-rejection test *is* the 003417 mechanism.

**Files:**
- Modify: `src/makeup_transfer/priors.py`
- Modify: `tests/test_priors.py`

**Interfaces:**
- Consumes: `MakeupPrior`, `face_mask_flat`, `PRIOR_SIZE` (Task 2).
- Produces:
  - `nuisance_basis(size) -> np.ndarray` — (D, 18): per-channel (constant, x, y, x², xy, y²), block-diagonal across the 3 Lab channels.
  - `fit(prior, d_obs, *, lam=1.0, iters=3) -> FitResult` where `d_obs` is (D,) float32 and `FitResult` is a dataclass with `makeup` (D,), `nuisance` (D,), `c` (K,), `k` (18,).
  - `lab_diff_uv(uv_with, uv_without, size) -> np.ndarray` — (D,) raw face-masked Lab difference at `size`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_priors.py`:

```python
from makeup_transfer.priors import fit, nuisance_basis


def _toy_prior(size=PRIOR_SIZE, k=6, seed=0):
    """A prior whose components are smooth-ish random fields — stands in for real makeup."""
    d = 3 * int(face_mask_flat(size).sum())
    rng = np.random.default_rng(seed)
    comps = rng.normal(size=(k, d)).astype(np.float32)
    comps /= np.linalg.norm(comps, axis=1, keepdims=True)
    return MakeupPrior(
        mean=np.zeros(d, dtype=np.float32),
        components=comps,
        variances=np.linspace(4.0, 1.0, k).astype(np.float32),
        size=size,
    )


def test_nuisance_basis_spans_constant_and_gradient():
    n = nuisance_basis(PRIOR_SIZE)
    d = 3 * int(face_mask_flat(PRIOR_SIZE).sum())
    assert n.shape == (d, 18)
    # A per-channel constant offset must be exactly representable.
    offset = np.zeros(d, dtype=np.float32)
    third = d // 3
    offset[:third] = 2.5                      # L channel shifted
    coef, *_ = np.linalg.lstsq(n, offset, rcond=None)
    np.testing.assert_allclose(n @ coef, offset, atol=1e-4)


def test_basis_roundtrip_recovers_known_component():
    """A difference synthesized from the basis is recovered by the fit."""
    prior = _toy_prior()
    truth = 2.0 * prior.components[1]
    res = fit(prior, truth.astype(np.float32), lam=1e-6)
    np.testing.assert_allclose(res.makeup, truth, atol=0.05)


def test_nuisance_rejection_is_the_003417_fix():
    """ZERO makeup + a global colour offset + a smooth gradient => recovered makeup ~= 0.

    This is exactly the SSAT skin-tone-drift failure. A raw chromaticity difference reports the
    whole face as makeup here; the prior fit must not.
    """
    prior = _toy_prior()
    size = PRIOR_SIZE
    mask = face_mask_flat(size)
    yy, xx = np.mgrid[0:size, 0:size]
    xn = ((xx / (size - 1)) * 2 - 1).reshape(-1)[mask]
    yn = ((yy / (size - 1)) * 2 - 1).reshape(-1)[mask]

    # per-channel constant drift + a smooth left-right shading gradient — no makeup at all
    drift = np.concatenate([
        1.8 + 0.9 * xn,          # L
        2.4 + 0.6 * yn,          # a
        -1.5 + 0.4 * xn,         # b
    ]).astype(np.float32)

    res = fit(prior, drift, lam=1.0)

    assert np.abs(res.makeup).mean() < 0.05 * np.abs(drift).mean(), (
        "prior fit absorbed skin-tone drift as makeup — the 003417 bug"
    )
    np.testing.assert_allclose(res.nuisance, drift, atol=0.1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_priors.py -v`
Expected: FAIL — `ImportError: cannot import name 'fit'`

- [ ] **Step 3: Implement the basis and the fit**

Append to `src/makeup_transfer/priors.py`:

```python
@dataclass
class FitResult:
    makeup: np.ndarray     # (D,) mean + M.c   — what we keep
    nuisance: np.ndarray   # (D,) N.k          — what we throw away
    c: np.ndarray          # (K,) makeup coefficients
    k: np.ndarray          # (18,) nuisance coefficients


def nuisance_basis(size: int = PRIOR_SIZE) -> np.ndarray:
    """Smooth-nuisance model: per-channel constant + degree-2 polynomial in UV coords.

    This is the explicit model of shading, exposure, white balance and skin-tone drift — the
    things a *difference* wrongly calls makeup. Block-diagonal over the 3 Lab channels, so each
    channel drifts independently (a warm skin-tone shift moves a/b but barely L).
    """
    mask = face_mask_flat(size)
    yy, xx = np.mgrid[0:size, 0:size]
    x = ((xx / (size - 1)) * 2 - 1).reshape(-1)[mask]
    y = ((yy / (size - 1)) * 2 - 1).reshape(-1)[mask]
    terms = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)  # (P, 6)

    p = terms.shape[0]
    basis = np.zeros((3 * p, 18), dtype=np.float32)
    for ch in range(3):
        basis[ch * p:(ch + 1) * p, ch * 6:(ch + 1) * 6] = terms
    return basis


def fit(prior: MakeupPrior, d_obs: np.ndarray, *, lam: float = 1.0, iters: int = 3) -> FitResult:
    """Robustly explain `d_obs` as makeup + nuisance, and hand back the two parts separately.

    `lam` scales a ridge on the makeup coefficients weighted by 1/variance — implausible makeup is
    penalized in proportion to how implausible the basis says it is. That shrinkage is what makes
    this a *prior* rather than a bare projection. The nuisance coefficients are left unpenalized:
    we want the nuisance to soak up drift greedily, since it gets discarded anyway.

    IRLS with Huber weights, so a few outlier pixels (a stray specular, a glasses rim) cannot drag
    the whole fit.
    """
    m = prior.components.T.astype(np.float32)          # (D, K)
    n = nuisance_basis(prior.size)                     # (D, 18)
    a = np.concatenate([m, n], axis=1)                 # (D, K+18)
    target = (d_obs - prior.mean).astype(np.float32)

    kdim = m.shape[1]
    penalty = np.zeros(a.shape[1], dtype=np.float32)
    penalty[:kdim] = lam / np.maximum(prior.variances, 1e-6)
    reg = np.diag(penalty)

    w = np.ones(a.shape[0], dtype=np.float32)
    coef = np.zeros(a.shape[1], dtype=np.float32)
    for _ in range(iters):
        aw = a * w[:, None]
        coef = np.linalg.solve(aw.T @ a + reg, aw.T @ target)
        resid = np.abs(target - a @ coef)
        # Huber: unit weight inside the scale, 1/r beyond it. Scale = a robust spread estimate.
        scale = max(1.4826 * float(np.median(resid)), 1e-3)
        w = np.minimum(1.0, scale / np.maximum(resid, 1e-6)).astype(np.float32)

    c, k = coef[:kdim], coef[kdim:]
    return FitResult(makeup=prior.mean + m @ c, nuisance=n @ k, c=c, k=k)


def lab_diff_uv(uv_with: np.ndarray, uv_without: np.ndarray, size: int = PRIOR_SIZE) -> np.ndarray:
    """RAW face-masked Lab difference at `size`, channels concatenated (L, a, b).

    Deliberately raw: no Retinex, no median a/b shift. Those Phase-1 corrections exist only to
    paper over illumination drift, and `nuisance_basis` is the principled version of them —
    stacking both would correct the same nuisance twice, invisibly to the fit.
    """
    import cv2

    from .extract import _rgb_lab

    small_w = cv2.resize(uv_with, (size, size), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(uv_without, (size, size), interpolation=cv2.INTER_AREA)
    d = _rgb_lab(small_w) - _rgb_lab(small_b)            # (size, size, 3)
    mask = face_mask_flat(size)
    return np.concatenate([d[..., ch].reshape(-1)[mask] for ch in range(3)]).astype(np.float32)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_priors.py -v`
Expected: PASS (6 tests). `test_nuisance_rejection_is_the_003417_fix` passing is the milestone — the mechanism works.

- [ ] **Step 5: Commit**

```bash
git add src/makeup_transfer/priors.py tests/test_priors.py
git commit -m "Phase 4: nuisance basis + robust IRLS fit (discarding nuisance rejects skin-tone drift)"
```

---

### Task 4: Corpus builder — learn the basis

**Files:**
- Modify: `src/makeup_transfer/priors.py`
- Modify: `tests/test_priors.py`

**Interfaces:**
- Consumes: `MakeupPrior`, `lab_diff_uv`, `PRIOR_SIZE`.
- Produces: `build_prior(pairs, *, size=PRIOR_SIZE, var_target=0.95, max_components=64) -> MakeupPrior`, where `pairs` is a list of `(with_path, without_path)`; `find_pairs(data_dir) -> list[tuple[Path, Path]]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_priors.py`:

```python
from makeup_transfer.priors import build_prior_from_diffs


def test_build_prior_from_diffs_recovers_a_planted_direction():
    """PCA over synthetic diffs that all lie along one direction => first component is it."""
    d = 3 * int(face_mask_flat(PRIOR_SIZE).sum())
    rng = np.random.default_rng(1)
    direction = rng.normal(size=d).astype(np.float32)
    direction /= np.linalg.norm(direction)

    amps = rng.normal(scale=3.0, size=40).astype(np.float32)
    diffs = np.stack([a * direction for a in amps])          # (40, D), rank 1

    prior = build_prior_from_diffs(diffs, var_target=0.95, max_components=8)

    assert prior.n_components >= 1
    # First component is the planted direction, up to sign.
    cos = abs(float(prior.components[0] @ direction))
    assert cos > 0.99
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_priors.py::test_build_prior_from_diffs_recovers_a_planted_direction -v`
Expected: FAIL — `ImportError: cannot import name 'build_prior_from_diffs'`

- [ ] **Step 3: Implement the builder**

Append to `src/makeup_transfer/priors.py`:

```python
def build_prior_from_diffs(
    diffs: np.ndarray,
    *,
    var_target: float = 0.95,
    max_components: int = 64,
    size: int = PRIOR_SIZE,
) -> MakeupPrior:
    """PCA over a (N, D) stack of difference fields.

    Samples (~2k) are far fewer than dimensions (~30k), so we eigendecompose the N x N Gram matrix
    and map back — same result as an SVD of the data matrix, a fraction of the cost, and no
    scikit-learn dependency.
    """
    x = diffs.astype(np.float32)
    mean = x.mean(axis=0)
    xc = x - mean

    gram = xc @ xc.T                                   # (N, N)
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    keep = vals > 1e-8
    vals, vecs = vals[keep], vecs[:, keep]
    if vals.size == 0:
        raise ValueError("degenerate corpus: no variance to build a basis from")

    ratio = np.cumsum(vals) / vals.sum()
    k = int(np.searchsorted(ratio, var_target) + 1)
    k = max(1, min(k, max_components, vals.size))

    comps = (xc.T @ vecs[:, :k]) / np.sqrt(np.maximum(vals[:k], 1e-12))   # (D, k), unit norm
    variances = (vals[:k] / max(x.shape[0] - 1, 1)).astype(np.float32)
    return MakeupPrior(
        mean=mean, components=comps.T.astype(np.float32), variances=variances, size=size,
    )


def find_pairs(data_dir: str | Path) -> list[tuple[Path, Path]]:
    """(makeup, bare) photo pairs from a directory of `<id>_bare.jpg` / `<id>_makeup_NN.jpg`."""
    data_dir = Path(data_dir)
    pairs: list[tuple[Path, Path]] = []
    for bare in sorted(data_dir.glob("*_bare.jpg")):
        ident = bare.name.split("_")[0]
        for makeup in sorted(data_dir.glob(f"{ident}_makeup_*.jpg")):
            pairs.append((makeup, bare))
    return pairs


def build_prior(
    pairs: list[tuple[Path, Path]],
    *,
    size: int = PRIOR_SIZE,
    var_target: float = 0.95,
    max_components: int = 64,
) -> MakeupPrior:
    """Unwrap every pair to UV, take the raw Lab difference, and PCA the stack.

    Pairs whose face detection fails are skipped and *counted* — a silent drop here would quietly
    bias the basis.
    """
    from .extract import unwrap_to_uv

    diffs, failed = [], 0
    for i, (makeup, bare) in enumerate(pairs, 1):
        try:
            _, uv_m, _ = unwrap_to_uv(makeup)
            _, uv_b, _ = unwrap_to_uv(bare)
        except Exception as exc:                       # noqa: BLE001 — report, don't hide
            failed += 1
            print(f"  skip {makeup.name}: {type(exc).__name__}: {exc}")
            continue
        diffs.append(lab_diff_uv(uv_m, uv_b, size))
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)} pairs unwrapped ({failed} failed)")

    if not diffs:
        raise ValueError("no usable pairs — check the data directory")
    print(f"Built {len(diffs)} difference fields ({failed} pairs failed face detection)")
    return build_prior_from_diffs(
        np.stack(diffs), var_target=var_target, max_components=max_components, size=size,
    )
```

- [ ] **Step 4: Add the `--build` CLI**

Append to `src/makeup_transfer/priors.py`:

```python
def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build the PCA makeup prior from an FFHQ-Makeup subset.")
    ap.add_argument("--build", action="store_true", help="build the prior and save it")
    ap.add_argument("--data", default="data/ffhq_makeup_train", help="directory of bare/makeup photos")
    ap.add_argument("--out", default=str(PRIOR_PATH))
    ap.add_argument("--size", type=int, default=PRIOR_SIZE)
    ap.add_argument("--var", type=float, default=0.95, help="explained-variance target")
    ap.add_argument("--max-components", type=int, default=64)
    args = ap.parse_args()

    if not args.build:
        ap.error("nothing to do: pass --build")

    pairs = find_pairs(args.data)
    print(f"Found {len(pairs)} pairs in {args.data}")
    prior = build_prior(
        pairs, size=args.size, var_target=args.var, max_components=args.max_components,
    )
    prior.save(args.out)
    print(f"Saved prior: K={prior.n_components} components, size={prior.size} -> {args.out}")


if __name__ == "__main__":
    _main()
```

- [ ] **Step 5: Run the test, then build a real prior**

```bash
python -m pytest tests/test_priors.py -v
```
Expected: PASS (7 tests).

Then build a **smoke-test prior** from whatever data is on hand (the 3 sample identities are held-out, so this is only to prove the path runs end to end — it is not the real basis):

```bash
PYTHONPATH=src python -m makeup_transfer.priors --build --data data/ffhq_makeup --out weights/prior_smoke.npz --max-components 8
```
Expected: prints pair count, unwrap progress, and `Saved prior: K=... components`.

Once Task 1's corpus exists, build the real one:

```bash
PYTHONPATH=src python -m makeup_transfer.priors --build --data data/ffhq_makeup_train
```

- [ ] **Step 6: Commit**

```bash
git add src/makeup_transfer/priors.py tests/test_priors.py
git commit -m "Phase 4: PCA corpus builder (Gram-matrix trick, no sklearn) + --build CLI"
```

---

### Task 5: `extract_layer_prior` — alpha from the prior, plus the gated residual

**Files:**
- Modify: `src/makeup_transfer/priors.py`
- Modify: `tests/test_priors.py`

**Interfaces:**
- Consumes: everything above; `extract.ALPHA_LO`, `extract.ALPHA_HI`, `extract._smoothstep`, `extract._rgb_lab`; `uv_unwrap.region_masks_uv`.
- Produces: `extract_layer_prior(uv_with, vis_with, uv_without, vis_without, *, prior, residual=0.0, lam=1.0, sym=0.0) -> (layer_rgba, skin_lab_mean, debug)` — the **same return contract** as `extract.extract_layer`, so it is a drop-in swap.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_priors.py`:

```python
import cv2

from makeup_transfer.priors import extract_layer_prior, unflatten


def test_unflatten_roundtrip():
    size = PRIOR_SIZE
    mask = face_mask_flat(size)
    d = 3 * int(mask.sum())
    vec = np.arange(d, dtype=np.float32)
    img = unflatten(vec, size)
    assert img.shape == (size, size, 3)
    # outside the face chart it must be zero, inside it must carry the values back
    assert float(np.abs(img.reshape(-1, 3)[~mask]).max()) == 0.0
    np.testing.assert_allclose(img[..., 0].reshape(-1)[mask], vec[: int(mask.sum())])


def _smooth_basis_prior(size=PRIOR_SIZE, k=4, seed=7):
    """A prior whose components are genuinely LOW-FREQUENCY, so a thin stroke provably
    cannot be represented by them. This is what makes the residual test meaningful."""
    mask = face_mask_flat(size)
    rng = np.random.default_rng(seed)
    comps = []
    for _ in range(k):
        f = cv2.GaussianBlur(rng.normal(size=(size, size)).astype(np.float32), (0, 0), 12.0)
        v = np.concatenate([f.reshape(-1)[mask]] * 3)
        comps.append(v / np.linalg.norm(v))
    d = 3 * int(mask.sum())
    return MakeupPrior(
        mean=np.zeros(d, dtype=np.float32),
        components=np.stack(comps).astype(np.float32),
        variances=np.linspace(4.0, 1.0, k).astype(np.float32),
        size=size,
    )


def _stroke_uv_pair(size=512, stroke_size=PRIOR_SIZE):
    """A UV 'pair' differing only by one thin bright stroke on the lips region."""
    uv_without = np.full((size, size, 3), 150, dtype=np.uint8)
    uv_with = uv_without.copy()
    scale = size // stroke_size
    cv2.line(uv_with, (40 * scale, 64 * scale), (88 * scale, 64 * scale), (40, 40, 220), 2 * scale)
    vis = np.ones((size, size), dtype=np.float32)
    return uv_with, vis, uv_without, vis


def test_prior_only_cannot_represent_a_thin_stroke():
    """w=0: a smooth basis must NOT capture high-frequency detail. Half of the residual story."""
    prior = _smooth_basis_prior()
    mask = face_mask_flat(prior.size)
    stroke = np.zeros((prior.size, prior.size), dtype=np.float32)
    cv2.line(stroke, (40, 64), (88, 64), 12.0, 1)
    d_obs = np.concatenate([stroke.reshape(-1)[mask]] * 3).astype(np.float32)

    recovered = fit(prior, d_obs, lam=1.0).makeup
    assert float(np.abs(recovered).max()) < 0.5 * float(np.abs(d_obs).max())


def test_residual_recovers_high_frequency_detail_the_basis_cannot():
    """The residual knob's whole reason to exist: w>0 must put back detail w=0 loses.

    Drives extract_layer_prior end to end (this is its unit coverage), on a UV pair whose only
    difference is a thin stroke the smooth basis provably cannot represent.
    """
    prior = _smooth_basis_prior()
    uv_with, vis_w, uv_without, vis_b = _stroke_uv_pair()

    _, _, dbg_off = extract_layer_prior(
        uv_with, vis_w, uv_without, vis_b, prior=prior, residual=0.0)
    _, _, dbg_on = extract_layer_prior(
        uv_with, vis_w, uv_without, vis_b, prior=prior, residual=1.0)

    # The stroke sits inside the face chart; measure alpha along it.
    s = uv_with.shape[0]
    band = np.zeros((s, s), dtype=bool)
    scale = s // prior.size
    band[64 * scale - scale:64 * scale + scale, 40 * scale:88 * scale] = True

    assert dbg_on["alpha"][band].mean() > dbg_off["alpha"][band].mean(), (
        "residual=1.0 must recover more of the stroke than residual=0.0"
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_priors.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_layer_prior'`

- [ ] **Step 3: Implement**

Append to `src/makeup_transfer/priors.py`:

```python
def unflatten(vec: np.ndarray, size: int = PRIOR_SIZE) -> np.ndarray:
    """Inverse of the face-masked flatten: (D,) -> (size, size, 3), zero outside the face."""
    mask = face_mask_flat(size)
    p = int(mask.sum())
    img = np.zeros((size * size, 3), dtype=np.float32)
    for ch in range(3):
        img[mask, ch] = vec[ch * p:(ch + 1) * p]
    return img.reshape(size, size, 3)


def extract_layer_prior(
    uv_with: np.ndarray,
    vis_with: np.ndarray,
    uv_without: np.ndarray,
    vis_without: np.ndarray,
    *,
    prior: MakeupPrior,
    residual: float = 0.0,
    lam: float = 1.0,
    sym: float = 0.0,
):
    """Prior-fitted replacement for ``extract.extract_layer``. Same return contract.

    ``residual`` (the ``w`` knob) adds back the high-frequency part of what the fit could not
    explain, gated to where the prior says makeup actually is. ``residual=0`` is prior-only.
    ``sym`` defaults to 0 here: shadow rejection is the prior's job now, not symmetry's.
    """
    import cv2

    from .extract import ALPHA_HI, ALPHA_LO, _rgb_lab, _smoothstep
    from .uv_unwrap import region_masks_uv

    s = uv_with.shape[0]
    px = s // 100
    size = prior.size

    d_obs = lab_diff_uv(uv_with, uv_without, size)
    res = fit(prior, d_obs, lam=lam, iters=3)

    makeup = res.makeup
    if residual > 0.0:
        r = d_obs - res.makeup - res.nuisance
        r_img = unflatten(r, size)
        r_hi = r_img - cv2.GaussianBlur(r_img, (0, 0), max(size / 64.0, 1.0))

        # Support gate: only where the prior already believes makeup lives.
        m_img = unflatten(res.makeup, size)
        support = np.linalg.norm(m_img, axis=2)
        g = _smoothstep(support, ALPHA_LO * 0.5, ALPHA_HI * 0.5)
        g = cv2.GaussianBlur(g, (0, 0), max(size / 64.0, 1.0))

        gated = r_hi * (residual * g)[..., None]
        mask = face_mask_flat(size)
        makeup = makeup + np.concatenate(
            [gated[..., ch].reshape(-1)[mask] for ch in range(3)]
        ).astype(np.float32)

    makeup_img = unflatten(makeup, size)
    makeup_full = cv2.resize(makeup_img, (s, s), interpolation=cv2.INTER_LINEAR)

    # Alpha from the magnitude of the reconstructed makeup field, same ramp as the baseline.
    dist = np.linalg.norm(makeup_full, axis=2)
    alpha = _smoothstep(dist, ALPHA_LO, ALPHA_HI)

    # --- gates: identical to the baseline extractor -----------------------------------------
    masks = region_masks_uv(s)
    gate = (masks["face"] & ~(masks["left_eye"] | masks["right_eye"]) & ~masks["mouth"])
    gate = cv2.GaussianBlur(gate.astype(np.float32) / 255.0, (0, 0), float(px))
    alpha *= gate
    alpha *= np.clip(np.minimum(vis_with, vis_without) * 2.0, 0.0, 1.0)

    if sym > 0.0:
        alpha = (1.0 - sym) * alpha + sym * np.minimum(alpha, alpha[:, ::-1])

    alpha = cv2.GaussianBlur(alpha, (0, 0), float(px))
    hard_excl = (masks["mouth"] | masks["left_eye"] | masks["right_eye"]) > 0
    alpha[hard_excl] = 0.0

    # Bare-skin estimate from the REAL photo: observed makeup Lab minus the fitted makeup field.
    # This is why the prior also fixes the apply side — skin_lab_mean no longer depends on the
    # pseudo-bare's absolute skin tone at all.
    lab_m = _rgb_lab(uv_with)
    bare_est = lab_m - makeup_full
    face = masks["face"] > 0
    skin_lab_mean = bare_est[face].mean(axis=0)

    rgb = cv2.cvtColor(uv_with, cv2.COLOR_BGR2RGB)
    layer = np.dstack([rgb, (alpha * 255).astype(np.uint8)])

    debug = {
        "alpha": alpha,
        "makeup_field": np.clip(np.abs(makeup_full) / ALPHA_HI, 0, 1),
        "nuisance_field": np.clip(
            np.abs(cv2.resize(unflatten(res.nuisance, size), (s, s))) / ALPHA_HI, 0, 1),
        "c": res.c,
        "k": res.k,
    }
    return layer, skin_lab_mean, debug
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_priors.py -v`
Expected: PASS (10 tests). `test_residual_recovers_high_frequency_detail_the_basis_cannot` passing is what proves the `w` knob does the job it exists for.

- [ ] **Step 5: Commit**

```bash
git add src/makeup_transfer/priors.py tests/test_priors.py
git commit -m "Phase 4: extract_layer_prior — prior alpha, gated residual, drift-free skin_lab_mean"
```

---

### Task 6: Wire the prior into `extract.py`

**Files:**
- Modify: `src/makeup_transfer/extract.py` (`extract_look` at lines 145-182, `extract_look_single` at 185-209, `_main` at 223-277)
- Test: `tests/test_extract_prior.py`

**Interfaces:**
- Consumes: `priors.MakeupPrior`, `priors.extract_layer_prior`, `priors.PRIOR_PATH`.
- Produces: `extract_look(..., prior=None, residual=0.0, lam=1.0)` and `extract_look_single(..., prior=None, residual=0.0, lam=1.0)`, where `prior` is a `MakeupPrior | str | Path | None`. `None` keeps today's baseline behaviour byte-for-byte.

- [ ] **Step 1: Write the failing test**

`tests/test_extract_prior.py`:

```python
"""Integration: prior-based extraction on a real FFHQ pair, against a tiny fixture prior."""
from pathlib import Path

import numpy as np
import pytest

from makeup_transfer.extract import extract_look, unwrap_to_uv
from makeup_transfer.priors import build_prior_from_diffs, lab_diff_uv
from makeup_transfer.storage import LookStore
from makeup_transfer.uv_unwrap import region_masks_uv

DATA = Path("data/ffhq_makeup")
MAKEUP = DATA / "002386_makeup_03.jpg"
BARE = DATA / "002386_bare.jpg"

if not MAKEUP.exists():
    pytest.skip("FFHQ-Makeup samples not present", allow_module_level=True)


@pytest.fixture(scope="module")
def tiny_prior():
    """A 3-component prior from the sample pairs. Not a real basis — just enough to run."""
    diffs = []
    for name in ["002386_makeup_01.jpg", "002386_makeup_03.jpg", "002386_makeup_05.jpg"]:
        _, uv_m, _ = unwrap_to_uv(DATA / name)
        _, uv_b, _ = unwrap_to_uv(BARE)
        diffs.append(lab_diff_uv(uv_m, uv_b))
    return build_prior_from_diffs(np.stack(diffs), max_components=3)


def test_extract_look_with_prior_puts_alpha_on_the_lips(tmp_path, tiny_prior):
    store = LookStore(root=tmp_path / "looks")
    layer, skin_lab, debug = extract_look(
        MAKEUP, BARE, "prior-test", store=store, prior=tiny_prior,
    )

    assert layer.shape[2] == 4
    alpha = debug["alpha"]
    masks = region_masks_uv(alpha.shape[0])
    lips = masks["lips"] > 0
    face = masks["face"] > 0

    # Lipstick is the strongest signal in this look: lip alpha must clearly exceed face-average.
    assert alpha[lips].mean() > 2.0 * alpha[face].mean()
    assert len(skin_lab) == 3
    assert store.load_meta("prior-test").blend_params["skin_lab_mean"]


def test_prior_none_is_byte_for_byte_the_baseline(tmp_path):
    """The invariant that protects every existing look: no prior => today's extractor, exactly.

    Compares extract_look(prior=None) against extract_layer called directly on the same UV pair.
    """
    from makeup_transfer.extract import extract_layer

    store = LookStore(root=tmp_path / "looks")
    layer, _, _ = extract_look(MAKEUP, BARE, "baseline", store=store)

    _, uv_m, vis_m = unwrap_to_uv(MAKEUP)
    _, uv_b, vis_b = unwrap_to_uv(BARE)
    expected, _, _ = extract_layer(uv_m, vis_m, uv_b, vis_b, sym=0.5)

    np.testing.assert_array_equal(layer, expected)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_extract_prior.py -v`
Expected: FAIL — `TypeError: extract_look() got an unexpected keyword argument 'prior'`

- [ ] **Step 3: Wire it into `extract_look`**

In `src/makeup_transfer/extract.py`, replace the signature and body of `extract_look` (currently lines 145-182) with:

```python
def extract_look(
    with_image: str | Path,
    without_image: str | Path,
    look_name: str,
    *,
    uv_size: int = UV_SIZE,
    sym: float = 0.5,
    store: LookStore | None = None,
    source_mode: str = "pair",
    prior=None,
    residual: float = 0.0,
    lam: float = 1.0,
):
    """Full extraction: photo pair -> UV pair -> RGBA layer -> saved look.

    ``prior`` (a ``MakeupPrior``, or a path to one) switches the alpha from the Phase-1
    chromaticity difference to the Phase-4 prior fit. ``None`` keeps the baseline unchanged.

    Returns ``(layer_rgba, skin_lab_mean, debug)`` (also persisted via the store).
    """
    from .apply import DEFAULT_REGION_PARAMS

    _, uv_m, vis_m = unwrap_to_uv(with_image, uv_size)
    _, uv_b, vis_b = unwrap_to_uv(without_image, uv_size)

    if prior is None:
        layer, skin_lab, debug = extract_layer(uv_m, vis_m, uv_b, vis_b, sym=sym)
        method = "difference"
    else:
        from .priors import MakeupPrior, extract_layer_prior

        if not isinstance(prior, MakeupPrior):
            prior = MakeupPrior.load(prior)
        # sym defaults OFF under the prior: rejecting shadow is the prior's job now.
        layer, skin_lab, debug = extract_layer_prior(
            uv_m, vis_m, uv_b, vis_b, prior=prior, residual=residual, lam=lam, sym=0.0,
        )
        method = "prior"

    store = store or LookStore()
    store.save(
        look_name,
        layer,
        meta={
            "source_with": str(with_image),
            "source_without": str(without_image),
            "source_mode": source_mode,
            "blend": "per-region",
            "blend_params": {
                "skin_lab_mean": [round(float(v), 2) for v in skin_lab],
                "regions": {k: dict(v) for k, v in DEFAULT_REGION_PARAMS.items()},
                # Nested here, NOT a top-level meta key: LookMeta.from_dict silently drops
                # unknown top-level fields (storage.py:63), so `extract_method` would vanish on
                # load without erroring. blend_params is a free-form dict, so it round-trips.
                "extract_method": method,
            },
        },
    )
    debug["uv_with"] = uv_m
    debug["uv_without"] = uv_b
    return layer, skin_lab, debug
```

**Verified 2026-07-14:** `LookMeta.from_dict` filters to known fields (`storage.py:63`), so a new top-level `extract_method` key would be dropped on load *silently*. It lives inside `blend_params` for that reason, and `storage.py` stays untouched per the Global Constraints.

- [ ] **Step 4: Thread the same params through `extract_look_single`**

Replace `extract_look_single`'s signature and its `return extract_look(...)` call (currently lines 185-209):

```python
def extract_look_single(
    with_image: str | Path,
    look_name: str,
    *,
    uv_size: int = UV_SIZE,
    sym: float = 0.5,
    store: LookStore | None = None,
    reference: str | Path | None = None,
    pseudo_out: str | Path | None = None,
    prior=None,
    residual: float = 0.0,
    lam: float = 1.0,
):
    """Single-photo extraction: SSAT removal synthesizes the bare face (Phase 2.5),
    then the ordinary pair pipeline runs on the pseudo-pair. The pseudo-bare is
    written to ``pseudo_out`` (or next to the phase-2.5 debug output) so it can be
    eyeballed and re-used.

    This is the mode the prior exists for: SSAT drifts the pseudo-bare's skin tone, and the
    prior's nuisance basis is what stops that drift being read as makeup.
    """
    from .removal import estimate_bare

    pseudo = estimate_bare(with_image, reference=reference)
    pseudo_path = Path(pseudo_out) if pseudo_out else (
        Path("preprocess_out/phase25") / f"{look_name}_pseudo_bare.png")
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pseudo_path), pseudo)
    return extract_look(
        with_image, pseudo_path, look_name,
        uv_size=uv_size, sym=sym, store=store, source_mode="single",
        prior=prior, residual=residual, lam=lam,
    )
```

- [ ] **Step 5: Add the CLI flags**

At the top of `_main()`, alongside the existing `import argparse`:

```python
    from .priors import PRIOR_PATH
```

Then, after the `--sym` argument, add:

```python
    ap.add_argument("--prior", nargs="?", const=str(PRIOR_PATH), default=None,
                    help="use the PCA makeup prior (bare flag = the default prior, "
                         "or pass a path to a prior .npz)")
    ap.add_argument("--residual", type=float, default=0.0,
                    help="gated high-frequency residual weight w in [0,1] (0 = prior only)")
    ap.add_argument("--prior-lambda", type=float, default=1.0,
                    help="ridge on the makeup coefficients (higher = stronger prior)")
```

Then pass them through both call sites in `_main()`:

```python
    if args.without_image is None:
        layer, skin_lab, debug = extract_look_single(
            args.with_image, args.look, uv_size=args.size, sym=args.sym,
            reference=args.ref,
            pseudo_out=out / f"{args.look}_pseudo_bare.png",
            prior=args.prior, residual=args.residual, lam=args.prior_lambda,
        )
        print(f"Single-photo mode: pseudo-bare written to {out / f'{args.look}_pseudo_bare.png'}")
    else:
        layer, skin_lab, debug = extract_look(
            args.with_image, args.without_image, args.look,
            uv_size=args.size, sym=args.sym,
            prior=args.prior, residual=args.residual, lam=args.prior_lambda,
        )
```

The debug-map writes at the end of `_main()` reference `debug["d_ab"]` and `debug["d_dark"]`, which the **prior path does not produce**. Guard them:

```python
    for name in ("d_ab", "d_dark", "makeup_field", "nuisance_field"):
        if name in debug:
            cv2.imwrite(str(out / f"{args.look}_{name}.png"),
                        (np.clip(debug[name], 0, 1) * 255).astype(np.uint8))
```

replacing the two unconditional `d_ab` / `d_dark` `imwrite` lines.

- [ ] **Step 6: Run the tests**

```bash
python -m pytest tests/test_extract_prior.py tests/test_priors.py -v
python -m pytest -q          # the whole suite — nothing else may regress
```
Expected: all pass, including the pre-existing `test_removal.py` and `test_extract_single.py`.

- [ ] **Step 7: Verify the CLI end-to-end**

```bash
PYTHONPATH=src python -m makeup_transfer.extract \
  data/ffhq_makeup/002386_makeup_03.jpg data/ffhq_makeup/002386_bare.jpg \
  --look prior-cli --prior weights/prior_smoke.npz \
  --apply-to data/ffhq_makeup/003417_bare.jpg
```
Expected: writes `preprocess_out/phase1/prior-cli_alpha.png`, `..._makeup_field.png`, `..._nuisance_field.png`, `..._result.png`. Open the alpha and result — they should be recognisably makeup, not noise.

- [ ] **Step 8: Commit**

```bash
git add src/makeup_transfer/extract.py tests/test_extract_prior.py
git commit -m "Phase 4: --prior extraction path wired into extract.py (baseline unchanged by default)"
```

---

### Task 7: The gate — `priors_eval.py`

**Files:**
- Create: `src/makeup_transfer/priors_eval.py`

**Interfaces:**
- Consumes: `extract_look`, `apply_look`, `BareFaceEstimator`, `MakeupPrior`, `region_masks_uv`.

Requires the real prior from Task 4 (`weights/makeup_prior_pca.npz`) and a CUDA GPU (SSAT).

- [ ] **Step 1: Write the eval harness**

`src/makeup_transfer/priors_eval.py`:

```python
"""Phase 4 gate: does the PCA prior beat the raw-difference baseline?

Mirrors phase25_spike.py. For each held-out identity, extract with the true bare (the reference)
and with the SSAT pseudo-bare, under the baseline and under the prior at several residual weights.
The metric that matters is **false-alpha in single mode**: makeup claimed where the ground-truth
pair says there is none. That is the 003417 skin-tone-drift failure, quantified.

Gate (design doc): 003417 false-alpha (baseline 0.1734) drops to roughly the level of the other
two (~0.02-0.04), with no regression on 002386 (0.0221) or 003125 (0.0389).

Run::

    PYTHONPATH=src python -m makeup_transfer.priors_eval
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .apply import apply_look
from .extract import extract_look
from .priors import PRIOR_PATH, MakeupPrior
from .removal import BareFaceEstimator
from .uv_unwrap import region_masks_uv

DATA = Path("data/ffhq_makeup")
OUT = Path("preprocess_out/phase4")

CASES = [
    ("002386", "002386_makeup_03.jpg", "003417_bare.jpg"),
    ("003125", "003125_makeup_02.jpg", "002386_bare.jpg"),
    ("003417", "003417_makeup_05.jpg", "003125_bare.jpg"),
]

RESIDUAL_SWEEP = [0.0, 0.25, 0.5, 1.0]

# Phase-2.5 baseline numbers, for the record (mean|dA| / false-alpha / missed).
BASELINE = {
    "002386": (0.1105, 0.0221, 0.0507),
    "003125": (0.0987, 0.0389, 0.0214),
    "003417": (0.2106, 0.1734, 0.0041),
}


def _metrics(a_pair: np.ndarray, a_single: np.ndarray) -> tuple[float, float, float]:
    """Same three metrics as the Phase-2.5 spike, so the numbers are directly comparable."""
    face = region_masks_uv(a_pair.shape[0])["face"] > 0
    mad = float(np.abs(a_pair - a_single)[face].mean())
    false_alpha = float(((a_single > 0.3) & (a_pair < 0.1))[face].mean())
    missed = float(((a_pair > 0.3) & (a_single < 0.1))[face].mean())
    return mad, false_alpha, missed


def _panel(img: np.ndarray, h: int = 256) -> np.ndarray:
    w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _gray_panel(a: np.ndarray, h: int = 256) -> np.ndarray:
    g = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return _panel(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), h)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = MakeupPrior.load(PRIOR_PATH)
    print(f"Prior: K={prior.n_components} components at {prior.size}px\n")
    est = BareFaceEstimator()
    rows = []

    for ident, makeup_name, ref_name in CASES:
        makeup = DATA / makeup_name
        true_bare = DATA / f"{ident}_bare.jpg"
        ref = DATA / ref_name

        pseudo_path = OUT / f"{ident}_pseudo_bare.png"
        cv2.imwrite(str(pseudo_path), est.estimate_bare(makeup, reference=ref))

        print(f"--- {ident} "
              f"(baseline: mean|dA|={BASELINE[ident][0]:.4f} "
              f"false-alpha={BASELINE[ident][1]:.4f} missed={BASELINE[ident][2]:.4f})")

        panels = [_panel(cv2.imread(str(makeup))), _panel(cv2.imread(str(pseudo_path)))]
        for w in RESIDUAL_SWEEP:
            _, _, dbg_pair = extract_look(
                makeup, true_bare, f"p4-{ident}-pair-w{w}", prior=prior, residual=w)
            _, _, dbg_single = extract_look(
                makeup, pseudo_path, f"p4-{ident}-single-w{w}", prior=prior, residual=w)

            mad, false_alpha, missed = _metrics(dbg_pair["alpha"], dbg_single["alpha"])
            print(f"  w={w:<5} mean|dA|={mad:.4f}  false-alpha={false_alpha:.4f}  "
                  f"missed={missed:.4f}")
            panels.append(_gray_panel(dbg_single["alpha"]))

        applied, _ = apply_look(f"p4-{ident}-single-w0.0", ref)
        panels.append(_panel(applied))
        rows.append(np.hstack(panels))
        print()

    width = max(r.shape[1] for r in rows)
    grid = np.vstack([cv2.copyMakeBorder(r, 0, 8, 0, width - r.shape[1], cv2.BORDER_CONSTANT)
                      for r in rows])
    cv2.imwrite(str(OUT / "prior_grid.png"), grid)
    print("Columns: makeup | pseudo-bare | alpha(single) per w in "
          f"{RESIDUAL_SWEEP} | applied(w=0)\nWrote {OUT / 'prior_grid.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate**

```bash
PYTHONPATH=src python -m makeup_transfer.priors_eval
```

Expected output: per-identity metrics for each `w`, and `preprocess_out/phase4/prior_grid.png`.

**Read the numbers honestly.** The gate is:
- `003417` single-mode **false-alpha drops from 0.1734 to roughly 0.02–0.04**, and
- `002386` (0.0221) and `003125` (0.0389) do **not** regress, and
- the applied results in the grid hold up to the eye.

If `w > 0` reopens glasses ghosting on 003417 (false-alpha climbing with `w`), that is the risk the spec predicted — record it and pick the best `w`, or fall back to `w = 0`.

**If the gate fails, do not flip the default.** Report the numbers and stop. A failed gate is a real result, not a reason to tune until it passes.

- [ ] **Step 3: Commit the eval (whatever the outcome)**

```bash
git add src/makeup_transfer/priors_eval.py
git commit -m "Phase 4: prior-vs-baseline gate harness (residual sweep on held-out identities)"
```

---

### Task 8: Record the outcome and (only if the gate passed) flip the default

**Files:**
- Modify: `docs/BUILD_PLAN.md` (progress log, and the "Open questions" makeup-priors entry at line ~293)
- Modify: `README.md` (repo layout table)
- Modify: `src/makeup_transfer/extract.py` (only if the gate passed)

- [ ] **Step 1: Write the progress-log entry**

Append to the `## Progress log` section of `docs/BUILD_PLAN.md`, following the existing entries' style — **including their honesty about what failed**. Record: the measured metrics for every `w`, the chosen `w`, whether the gate passed, whether the residual reopened glasses ghosting, and the corpus size actually used.

- [ ] **Step 2: Update the "Open questions" entry**

The entry currently reads:

```
- makeup-priors: PCA prior (fast, low-freq) vs StyleGAN2 prior (high-freq, heavier) — start PCA.
```

Replace it with the settled verdict and the date, in the style of the SSAT entry above it.

- [ ] **Step 3: Add `priors.py` to the README repo layout**

In `README.md`, after the `apply.py` line:

```
  priors.py      [Phase 4] PCA makeup prior: fit instead of difference        ✅
```

- [ ] **Step 4: Flip the default — ONLY if the gate passed**

If and only if Task 7's gate passed, change `extract.py`'s CLI so the prior is the default and the baseline is opt-out (`--no-prior`), and note it in the progress log. **If the gate failed, skip this step entirely** and leave `--prior` opt-in.

- [ ] **Step 5: Run the full suite and commit**

```bash
python -m pytest -q
```
Expected: all pass.

```bash
git add docs/BUILD_PLAN.md README.md src/makeup_transfer/extract.py
git commit -m "Phase 4: record prior gate outcome in BUILD_PLAN + README"
```

---

## Self-Review

**Spec coverage:**
- Nuisance basis + discard → Task 3 ✅
- PCA basis from FFHQ corpus → Tasks 1, 4 ✅
- Gated high-frequency residual, `w=0` = prior-only → Task 5 ✅
- Drift-free `skin_lab_mean` from the bare estimate → Task 5 ✅
- Raw Lab difference (no Retinex/median shift) → Task 3 (`lab_diff_uv`) ✅
- Existing gates unchanged, 512 working res, 128 prior res → Task 5 ✅
- Opt-in `--prior`, flip default only on a passing gate → Tasks 6, 8 ✅
- No silent fallback; `FileNotFoundError` names the build command → Task 2 ✅
- Corpus build reports failed-detection count → Task 4 ✅
- Held-out identities excluded from training → Task 1 (asserted in Step 4) ✅
- All four tests from the spec → Tasks 2, 3, 4, 5, 6 ✅
- Gate metrics + comparison grid → Task 7 ✅

**Facts verified against the real artifacts while writing this plan (do not re-derive):**
- `data/FFHQ-Makeup.zip` is downloaded: 3.6 GB, 128,590 members. Layout is **one directory per identity** (`000000/bare.jpg`), so Task 1's extractor flattens to `<id>_bare.jpg`.
- HF's `images/` folder holds **only 20 identities** — which is why the zip is a hard dependency.
- `LookMeta.from_dict` (`storage.py:63`) **silently drops unknown top-level keys**, so `extract_method` is nested inside `blend_params`.
- `region_masks_uv` keys used here (`face`, `lips`, `left_eye`, `right_eye`, `mouth`) all exist in `preprocess.REGION_GROUPS`.
- No new pip dependency is introduced: PCA uses the numpy Gram-matrix trick, not scikit-learn.
