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
