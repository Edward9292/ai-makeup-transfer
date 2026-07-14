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
PRIOR_LAMBDA = 0.1         # calibrated ridge on the makeup coefficients (see `fit`)

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


def fit(prior: MakeupPrior, d_obs: np.ndarray, *, lam: float = PRIOR_LAMBDA, iters: int = 3) -> FitResult:
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
    lam: float = PRIOR_LAMBDA,
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

        # Support gate: only where the prior already believes makeup lives. Gated on the
        # MEAN-SUBTRACTED deviation (res.makeup - prior.mean), not on res.makeup itself:
        # res.makeup = prior.mean + M.c, and the real prior's mean has a median magnitude of
        # ~3.9 Lab even where the observed difference is exactly zero. Thresholding
        # `|res.makeup|` (whether against an absolute or self-relative scale) therefore
        # measures "prior mean + makeup", not "makeup" — on a bare face it opens almost fully,
        # and on the heaviest-makeup face it can end up *more* closed than on a bare one,
        # because the gate is really tracking the mean field's own contrast, anti-correlated
        # with actual makeup strength. `M.c` alone is in real Lab units for any properly-built
        # corpus (components are unit-norm, coefficients scale like sqrt(variance)), so an
        # absolute threshold on it is scale-correct without any per-field normalization.
        mask = face_mask_flat(size)
        m_img = unflatten(res.makeup - prior.mean, size)
        support = np.linalg.norm(m_img, axis=2)
        g = _smoothstep(support, ALPHA_LO * 0.5, ALPHA_HI * 0.5)
        g = cv2.GaussianBlur(g, (0, 0), max(size / 64.0, 1.0))

        gated = r_hi * (residual * g)[..., None]
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
            np.abs(cv2.resize(unflatten(res.nuisance, size), (s, s),
                               interpolation=cv2.INTER_LINEAR)) / ALPHA_HI, 0, 1),
        "c": res.c,
        "k": res.k,
    }
    return layer, skin_lab_mean, debug


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
