"""Phase 4 gate: does the PCA prior beat the raw-difference baseline?

Mirrors phase25_spike.py. For each held-out identity, extract with the true bare (the reference)
and with the SSAT pseudo-bare, under the prior at several residual weights. The metric that
matters is **false-alpha in single mode**: makeup claimed where the ground-truth pair says there
is none. That is the 003417 skin-tone-drift failure, quantified.

Gate (design doc): 003417 false-alpha (baseline 0.1734) drops to roughly the level of the other
two (~0.02-0.04), with no regression on 002386 (0.0221) or 003125 (0.0389).

The pseudo-bare is SSAT's output, and SSAT needs CUDA. On a CPU-only box we reuse the pseudo-bare
phase25_spike.py already cached: its CASES are identical to ours (same makeup photo, same
cross-identity reference), so the cached PNG *is* `estimate_bare(makeup, reference=ref)` for these
inputs — and it is the very image the BASELINE numbers below were measured against, which is what
makes the comparison apples-to-apples. We never synthesize a stand-in: no GPU and no cache is a
hard error, because a made-up pseudo-bare would silently invalidate the gate.

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
from .uv_unwrap import region_masks_uv

DATA = Path("data/ffhq_makeup")
OUT = Path("preprocess_out/phase4")
PSEUDO_CACHE = Path("preprocess_out/phase25")

# (identity, makeup photo, other-identity bare used as removal reference + apply target).
# Identical to phase25_spike.CASES — that is what lets us reuse its cached pseudo-bares.
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


def _pseudo_bare(ident: str, makeup: Path, ref: Path) -> Path:
    """SSAT's pseudo-bare for `makeup`, from the GPU if we have one, else from the spike's cache."""
    out_path = OUT / f"{ident}_pseudo_bare.png"
    try:
        from .removal import BareFaceEstimator

        est = BareFaceEstimator()
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        cached = PSEUDO_CACHE / f"{ident}_pseudo_bare.png"
        if not cached.exists():
            raise RuntimeError(
                f"SSAT unavailable ({exc}) and no cached pseudo-bare at {cached}. "
                "Run phase25_spike.py on a CUDA box first — the gate is meaningless without "
                "the real pseudo-bare."
            ) from exc
        print(f"  [SSAT unavailable: {exc}]\n  [reusing cached pseudo-bare {cached}]")
        cv2.imwrite(str(out_path), cv2.imread(str(cached)))
        return out_path

    cv2.imwrite(str(out_path), est.estimate_bare(makeup, reference=ref))
    return out_path


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
    rows = []

    for ident, makeup_name, ref_name in CASES:
        makeup = DATA / makeup_name
        true_bare = DATA / f"{ident}_bare.jpg"
        ref = DATA / ref_name

        print(f"--- {ident} "
              f"(baseline: mean|dA|={BASELINE[ident][0]:.4f} "
              f"false-alpha={BASELINE[ident][1]:.4f} missed={BASELINE[ident][2]:.4f})")
        pseudo_path = _pseudo_bare(ident, makeup, ref)

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
