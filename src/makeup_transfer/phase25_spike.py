"""Phase 2.5 spike: is (makeup, SSAT-pseudo-bare) extraction as good as a real pair?

For each held-out FFHQ-Makeup identity: synthesize the bare face from the makeup
photo alone (reference = a DIFFERENT identity's bare photo, so nothing leaks), run
the existing pair extraction with both the true and the pseudo bare, then compare
alphas and cross-apply both looks to the reference identity.

Gate (design doc): same regions covered / no large false alpha / plausible apply.

Run::

    PYTHONPATH=src python -m makeup_transfer.phase25_spike
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .apply import apply_look
from .extract import extract_look
from .removal import BareFaceEstimator
from .uv_unwrap import region_masks_uv

DATA = Path("data/ffhq_makeup")
OUT = Path("preprocess_out/phase25")

# (identity, makeup photo, other-identity bare used as removal reference + apply target)
CASES = [
    ("002386", "002386_makeup_03.jpg", "003417_bare.jpg"),
    ("003125", "003125_makeup_02.jpg", "002386_bare.jpg"),
    ("003417", "003417_makeup_05.jpg", "003125_bare.jpg"),
]


def _panel(img: np.ndarray, h: int = 256) -> np.ndarray:
    w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _gray_panel(a: np.ndarray, h: int = 256) -> np.ndarray:
    g = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return _panel(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), h)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    est = BareFaceEstimator()
    rows = []

    for ident, makeup_name, ref_name in CASES:
        makeup, true_bare, ref = DATA / makeup_name, DATA / f"{ident}_bare.jpg", DATA / ref_name

        pseudo = est.estimate_bare(makeup, reference=ref)
        pseudo_path = OUT / f"{ident}_pseudo_bare.png"
        cv2.imwrite(str(pseudo_path), pseudo)

        _, _, dbg_pair = extract_look(makeup, true_bare, f"p25-{ident}-pair")
        _, _, dbg_single = extract_look(makeup, pseudo_path, f"p25-{ident}-single")

        a_pair, a_single = dbg_pair["alpha"], dbg_single["alpha"]
        face = region_masks_uv(a_pair.shape[0])["face"] > 0
        mad = float(np.abs(a_pair - a_single)[face].mean())
        false_alpha = float(((a_single > 0.3) & (a_pair < 0.1))[face].mean())
        missed = float(((a_pair > 0.3) & (a_single < 0.1))[face].mean())
        print(f"{ident}: mean|dA|={mad:.4f}  false-alpha={false_alpha:.4f}  missed={missed:.4f}")

        applied_pair, _ = apply_look(f"p25-{ident}-pair", ref)
        applied_single, _ = apply_look(f"p25-{ident}-single", ref)

        row = np.hstack([
            _panel(cv2.imread(str(makeup))), _panel(cv2.imread(str(true_bare))),
            _panel(pseudo), _gray_panel(a_pair), _gray_panel(a_single),
            _panel(applied_pair), _panel(applied_single),
        ])
        cv2.imwrite(str(OUT / f"{ident}_row.png"), row)
        rows.append(row)

    width = max(r.shape[1] for r in rows)
    grid = np.vstack([cv2.copyMakeBorder(r, 0, 8, 0, width - r.shape[1], cv2.BORDER_CONSTANT)
                      for r in rows])
    cv2.imwrite(str(OUT / "spike_grid.png"), grid)
    print(f"Columns: makeup | true bare | PSEUDO bare | alpha(pair) | alpha(single) "
          f"| applied(pair) | applied(single)\nWrote {OUT / 'spike_grid.png'}")


if __name__ == "__main__":
    main()
