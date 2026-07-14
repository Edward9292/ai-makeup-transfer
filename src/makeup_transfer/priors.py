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


def fit(prior: MakeupPrior, d_obs: np.ndarray, *, lam: float = 0.1, iters: int = 3) -> FitResult:
    """Robustly explain `d_obs` as makeup + nuisance, and hand back the two parts separately.

    `lam` scales a ridge on the makeup coefficients weighted by (mean variance / component
    variance) — implausible makeup is penalized in proportion to how implausible the basis says it
    is, *relative to the corpus's own average component variance*. That normalization keeps `lam`
    scale-free: it means the same thing regardless of whether the underlying PCA eigenvalues happen
    to be O(1) or O(1e6). That shrinkage is what makes this a *prior* rather than a bare projection.
    The nuisance coefficients are left unpenalized: we want the nuisance to soak up drift greedily,
    since it gets discarded anyway.

    The default `lam=0.1` is calibrated so that leading (plausible) components — the ones the
    corpus's PCA says are the most common shapes of real makeup — pass through nearly intact, while
    low-variance (implausible) components are still strongly shrunk. Because the per-component
    penalty is `lam * mean(variances) / variances[j]`, how much a given component actually retains
    depends on the corpus's own variance spectrum shape (how steeply it decays), not on `lam` alone
    — so `lam` is swept against the real corpus's spectrum at the Phase-4 gate, not assumed correct
    from this synthetic calibration.

    IRLS with Huber weights, so a few outlier pixels (a stray specular, a glasses rim) cannot drag
    the whole fit.
    """
    m = prior.components.T.astype(np.float32)          # (D, K)
    n = nuisance_basis(prior.size)                     # (D, 18)
    a = np.concatenate([m, n], axis=1)                 # (D, K+18)
    target = (d_obs - prior.mean).astype(np.float32)

    kdim = m.shape[1]
    penalty = np.zeros(a.shape[1], dtype=np.float32)
    v = np.maximum(prior.variances, 1e-6)
    penalty[:kdim] = lam * (float(v.mean()) / v)
    reg = np.diag(penalty)

    w = np.ones(a.shape[0], dtype=np.float32)
    coef = np.zeros(a.shape[1], dtype=np.float32)
    for i in range(iters):
        aw = a * w[:, None]
        coef = np.linalg.solve(aw.T @ a + reg, aw.T @ target)
        if i < iters - 1:
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
