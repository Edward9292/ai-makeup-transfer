"""Phase 1: pair-based makeup extraction in UV space (``extract.py``).

Input: a with-makeup and a without-makeup photo of the *same person*. Both are unwrapped into
canonical UV space (``uv_unwrap``), so the difference between the two textures is pixel-aligned
regardless of pose/expression differences between the shots. From that difference we build a
translucent RGBA makeup layer (color = with-makeup texture, alpha = "how much makeup is here"),
saved as a look via ``storage.LookStore``.

Illumination handling (the #1 risk per BUILD_PLAN.md), cheap first cut:

- **Retinex-style L normalization** — luminance is divided by its heavily-blurred self on each
  texture, flattening smooth shading/exposure differences between the two shots while keeping
  mid/high-frequency darkening (eyeliner, lipstick depth). Broad soft makeup (blush, foundation)
  is expected to survive mostly through chroma, not luminance.
- **Chromaticity difference** — the color part of the matte comes from Lab a/b only, with a
  single global a/b offset (median over the face) subtracted to cancel white-balance drift
  without erasing local color changes.
- **UV symmetry suppression** — makeup is mostly left/right symmetric, cast shadows are not;
  the matte is pulled toward the min of itself and its mirror (weight ``--sym``).

The proper fix (fitting the makeup-priors PCA model instead of differencing) is a planned
upgrade; this module keeps the steps separable so that swap stays local.

Run::

    python -m makeup_transfer.extract data/ffhq_makeup/002386_makeup_03.jpg \
        data/ffhq_makeup/002386_bare.jpg --look ffhq-002386-03 \
        --apply-to data/ffhq_makeup/003417_bare.jpg
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .preprocess import detect_face
from .storage import LookStore
from .uv_unwrap import mirror_fill, region_masks_uv, unwrap

UV_SIZE = 512

# Lab-distance thresholds for the alpha ramp (same scale as phase0, pair diff is cleaner).
ALPHA_LO = 3.5
ALPHA_HI = 16.0
DARK_WEIGHT = 0.35    # luminance darkening (eyeliner, mascara, deep lipstick)
BRIGHT_WEIGHT = 0.15  # luminance brightening (highlighter, concealer) — weaker, noisier


def _rgb_lab(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR -> float32 Lab (L in [0,100], a/b roughly [-128,127])."""
    from skimage import color

    return color.rgb2lab(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def unwrap_to_uv(image_path: str | Path, uv_size: int = UV_SIZE):
    """detect -> unwrap -> mirror-fill one photo. Returns ``(face, uv_texture, visibility)``."""
    face = detect_face(image_path)
    uv_tex, vis = unwrap(face.image, face.landmarks, uv_size)
    uv_tex, vis = mirror_fill(uv_tex, vis)
    return face, uv_tex, vis


def _normalized_luminance(lab: np.ndarray, face_mask: np.ndarray, sigma: float) -> np.ndarray:
    """Retinex-style flattening: L divided by its blurred self, rescaled back to L units."""
    L = lab[..., 0]
    base = cv2.GaussianBlur(L, (0, 0), sigma)
    base = np.maximum(base, 1e-3)
    return L / base * float(base[face_mask].mean())


def extract_layer(
    uv_with: np.ndarray,
    vis_with: np.ndarray,
    uv_without: np.ndarray,
    vis_without: np.ndarray,
    *,
    sym: float = 0.5,
):
    """Build an RGBA makeup layer from an aligned UV texture pair.

    Returns ``(layer_rgba, skin_lab_mean, debug)``: the (S, S, 4) uint8 layer (RGB channel
    order), the mean Lab of the *real* bare skin (used by the apply-time blend), and a dict of
    intermediate maps for eyeballing.
    """
    s = uv_with.shape[0]
    px = s // 100  # ~1% of texture as a blur/morphology unit
    masks = region_masks_uv(s)
    face = masks["face"] > 0

    lab_m = _rgb_lab(uv_with)
    lab_b = _rgb_lab(uv_without)

    # --- chromaticity difference, white-balance drift removed globally --------------------
    d_ab_raw = lab_m[..., 1:] - lab_b[..., 1:]
    shift = np.median(d_ab_raw[face], axis=0)  # robust: most of the face is *not* makeup-shifted
    d_ab = np.linalg.norm(d_ab_raw - shift, axis=2)

    # --- Retinex-normalized luminance difference ------------------------------------------
    sigma = s / 5.0
    Ln_m = _normalized_luminance(lab_m, face, sigma)
    Ln_b = _normalized_luminance(lab_b, face, sigma)
    d_dark = np.clip(Ln_b - Ln_m, 0.0, None)    # makeup darker than bare skin
    d_bright = np.clip(Ln_m - Ln_b, 0.0, None)  # makeup brighter (highlighter)

    dist = d_ab + DARK_WEIGHT * d_dark + BRIGHT_WEIGHT * d_bright
    alpha = _smoothstep(dist, ALPHA_LO, ALPHA_HI)

    # --- gates: face chart, minus eye openings and mouth interior (teeth/tongue) ----------
    gate = (masks["face"] & ~(masks["left_eye"] | masks["right_eye"]) & ~masks["mouth"])
    gate = cv2.GaussianBlur(gate.astype(np.float32) / 255.0, (0, 0), float(px))
    alpha *= gate
    alpha *= np.clip(np.minimum(vis_with, vis_without) * 2.0, 0.0, 1.0)

    # --- shadow suppression via UV left/right symmetry -------------------------------------
    alpha = (1.0 - sym) * alpha + sym * np.minimum(alpha, alpha[:, ::-1])

    alpha = cv2.GaussianBlur(alpha, (0, 0), float(px))

    # Re-assert hard exclusions AFTER blurring: the blur bleeds lip alpha into the thin
    # mouth-interior sliver of the UV chart, and an open-mouthed target stretches that sliver
    # across the whole mouth in image space (tinted teeth). Same for the eye openings.
    hard_excl = (masks["mouth"] | masks["left_eye"] | masks["right_eye"]) > 0
    alpha[hard_excl] = 0.0

    skin_lab_mean = lab_b[face].mean(axis=0)
    rgb = cv2.cvtColor(uv_with, cv2.COLOR_BGR2RGB)
    layer = np.dstack([rgb, (alpha * 255).astype(np.uint8)])

    debug = {
        "d_ab": np.clip(d_ab / ALPHA_HI, 0, 1),
        "d_dark": np.clip(d_dark / ALPHA_HI, 0, 1),
        "alpha": alpha,
    }
    return layer, skin_lab_mean, debug


def extract_look(
    with_image: str | Path,
    without_image: str | Path,
    look_name: str,
    *,
    uv_size: int = UV_SIZE,
    sym: float = 0.5,
    store: LookStore | None = None,
):
    """Full extraction: photo pair -> UV pair -> RGBA layer -> saved look.

    Returns ``(layer_rgba, skin_lab_mean, debug)`` (also persisted via the store).
    """
    import json

    _, uv_m, vis_m = unwrap_to_uv(with_image, uv_size)
    _, uv_b, vis_b = unwrap_to_uv(without_image, uv_size)
    layer, skin_lab, debug = extract_layer(uv_m, vis_m, uv_b, vis_b, sym=sym)

    store = store or LookStore()
    store.save(
        look_name,
        layer,
        meta={
            "source_with": str(with_image),
            "source_without": str(without_image),
            "blend": "lab-delta",
            "notes": json.dumps({"skin_lab_mean": [round(float(v), 2) for v in skin_lab]}),
        },
    )
    debug["uv_with"] = uv_m
    debug["uv_without"] = uv_b
    return layer, skin_lab, debug


def _checker_preview(layer_rgba: np.ndarray, cell: int = 16) -> np.ndarray:
    """Composite the RGBA layer over a checkerboard so the alpha is eyeballable."""
    s = layer_rgba.shape[0]
    yy, xx = np.mgrid[0:s, 0:s]
    checker = np.where(((yy // cell + xx // cell) % 2) > 0, 200, 140).astype(np.float32)
    checker = np.dstack([checker] * 3)
    a = (layer_rgba[..., 3:4].astype(np.float32)) / 255.0
    bgr = cv2.cvtColor(layer_rgba[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    return ((1 - a) * checker + a * bgr).astype(np.uint8)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract a makeup layer from a with/without pair of the same person."
    )
    ap.add_argument("with_image", help="face WITH makeup")
    ap.add_argument("without_image", help="same face WITHOUT makeup")
    ap.add_argument("--look", default="extracted", help="look name to save")
    ap.add_argument("--out", default="preprocess_out/phase1", help="debug output directory")
    ap.add_argument("--size", type=int, default=UV_SIZE, help="UV texture size")
    ap.add_argument("--sym", type=float, default=0.5,
                    help="symmetry shadow-suppression weight in [0,1]")
    ap.add_argument("--apply-to", default=None,
                    help="optional target photo to re-apply the layer to (verification)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layer, skin_lab, debug = extract_look(
        args.with_image, args.without_image, args.look, uv_size=args.size, sym=args.sym
    )
    store = LookStore()
    print(f"Saved look '{args.look}' to {store.root / args.look}")

    cv2.imwrite(str(out / f"{args.look}_uv_with.png"), debug["uv_with"])
    cv2.imwrite(str(out / f"{args.look}_uv_without.png"), debug["uv_without"])
    cv2.imwrite(str(out / f"{args.look}_alpha.png"), (debug["alpha"] * 255).astype(np.uint8))
    cv2.imwrite(str(out / f"{args.look}_d_ab.png"), (debug["d_ab"] * 255).astype(np.uint8))
    cv2.imwrite(str(out / f"{args.look}_d_dark.png"), (debug["d_dark"] * 255).astype(np.uint8))
    cv2.imwrite(str(out / f"{args.look}_layer_preview.png"), _checker_preview(layer))

    if args.apply_to:
        from .phase0_spike import apply_layer  # Phase 2 replaces this with apply.py

        face_t = detect_face(args.apply_to)
        result = apply_layer(face_t.image, face_t.landmarks, layer, skin_lab)
        cv2.imwrite(str(out / f"{args.look}_result.png"), result)
        side = np.hstack([face_t.image, result])
        cv2.imwrite(str(out / f"{args.look}_side_by_side.png"), side)
        print(f"Applied to {args.apply_to} -> {out / f'{args.look}_result.png'}")

    print(f"Wrote debug maps to {out}/")


if __name__ == "__main__":
    _main()
