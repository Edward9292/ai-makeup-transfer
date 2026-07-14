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
