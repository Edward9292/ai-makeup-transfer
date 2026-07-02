"""Phase 0 feasibility spike: hand-derive a UV makeup layer and re-apply it to another face.

This is the go/no-go experiment from BUILD_PLAN.md, *not* the real extraction NN:

1. Unwrap a with-makeup face into canonical UV space (``uv_unwrap``).
2. Hand-derive a translucent RGBA layer: color from the makeup texture, alpha from the Lab
   distance to an inpainted "bare skin" estimate, gated to lip/eye regions.
3. Save it as a look (``storage.LookStore``).
4. Apply to a different person: skin-tone-aware Lab blend in UV space (target keeps its own
   luminance base, shifted by the makeup's *relative* darkening; chroma mixes toward the
   makeup color by alpha), then re-project to the target photo.

Run::

    python -m makeup_transfer.phase0_spike data/mt/makeup_vFG586.png data/mt/nomakeup_xfsy_0106.png
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .preprocess import detect_face
from .storage import LookStore
from .uv_unwrap import mirror_fill, project, region_masks_uv, unwrap

UV_SIZE = 512


def _rgb_lab(bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR -> float32 Lab (L in [0,100], a/b roughly [-128,127])."""
    from skimage import color

    return color.rgb2lab(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)


def _lab_bgr(lab: np.ndarray) -> np.ndarray:
    from skimage import color

    rgb = np.clip(color.lab2rgb(lab.astype(np.float64)), 0.0, 1.0)
    return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def derive_layer(uv_tex: np.ndarray, visibility: np.ndarray):
    """Hand-derive an RGBA makeup layer from a with-makeup UV texture.

    Returns ``(layer_rgba, skin_lab_mean)``: (S, S, 4) uint8 (RGB channel order, as stored),
    and the mean Lab of the estimated bare skin (needed by the blend at apply time).
    """
    s = UV_SIZE
    masks = region_masks_uv(s)
    px = s // 100  # ~1% of texture as a morphology unit

    lips = _dilate(masks["lips"], px)
    eyes = masks["left_eye"] | masks["right_eye"]
    brows = _dilate(masks["left_brow"] | masks["right_brow"], px)
    # Eyeshadow lives in a band around (mostly above) the eyes, between lids and brows.
    eye_band = (_dilate(eyes, 7 * px) & ~_dilate(eyes, px // 2) & ~brows)
    gate = ((lips | eye_band) & masks["face"]).astype(np.float32) / 255.0
    gate = cv2.GaussianBlur(gate, (0, 0), 2 * px)

    # Bare-skin estimate: inpaint the makeup-able regions away, then flatten with a big blur.
    excl = _dilate(lips | _dilate(eyes, 7 * px) | brows, px)
    skin = cv2.inpaint(uv_tex, excl, 3, cv2.INPAINT_TELEA)
    skin = cv2.GaussianBlur(skin, (0, 0), 3 * px)

    lab_tex = _rgb_lab(uv_tex)
    lab_skin = _rgb_lab(skin)
    d_ab = np.linalg.norm(lab_tex[..., 1:] - lab_skin[..., 1:], axis=2)
    d_l = np.abs(lab_tex[..., 0] - lab_skin[..., 0])
    dist = d_ab + 0.35 * d_l

    alpha = _smoothstep(dist, 4.0, 18.0) * gate * np.clip(visibility * 2.0, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), float(px))

    face_mask = masks["face"] > 0
    skin_lab_mean = lab_skin[face_mask].mean(axis=0)

    rgb = cv2.cvtColor(uv_tex, cv2.COLOR_BGR2RGB)
    layer = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    return layer, skin_lab_mean


def apply_layer(
    target_bgr: np.ndarray,
    target_landmarks: np.ndarray,
    layer_rgba: np.ndarray,
    skin_lab_mean: np.ndarray,
) -> np.ndarray:
    """Apply a UV makeup layer to a target photo with a skin-tone-aware Lab blend."""
    uv_b, _ = unwrap(target_bgr, target_landmarks, UV_SIZE)

    layer_bgr = cv2.cvtColor(layer_rgba[..., :3], cv2.COLOR_RGB2BGR)
    alpha = layer_rgba[..., 3].astype(np.float32) / 255.0

    lab_b = _rgb_lab(uv_b)
    lab_l = _rgb_lab(layer_bgr)
    out = lab_b.copy()
    # Luminance: keep the target's own base/texture, shift by the makeup's darkening
    # relative to the *source* skin -> the same look adapts to different skin tones.
    out[..., 0] = lab_b[..., 0] + alpha * (lab_l[..., 0] - float(skin_lab_mean[0]))
    out[..., 1] = (1 - alpha) * lab_b[..., 1] + alpha * lab_l[..., 1]
    out[..., 2] = (1 - alpha) * lab_b[..., 2] + alpha * lab_l[..., 2]
    out[..., 0] = np.clip(out[..., 0], 0.0, 100.0)
    blended_uv = _lab_bgr(out)

    h, w = target_bgr.shape[:2]
    blended_img, _ = project(blended_uv, target_landmarks, h, w)
    alpha_img, _ = project((alpha * 255).astype(np.uint8), target_landmarks, h, w)
    a = (alpha_img.astype(np.float32) / 255.0)[..., None]
    result = (1 - a) * target_bgr.astype(np.float32) + a * blended_img.astype(np.float32)
    return result.astype(np.uint8)


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
    import json

    ap = argparse.ArgumentParser(description="Phase 0 spike: derive a makeup layer, re-apply it.")
    ap.add_argument("makeup_image", help="face WITH makeup (layer source)")
    ap.add_argument("target_image", help="face WITHOUT makeup (apply target)")
    ap.add_argument("--look", default="phase0-spike", help="look name to save")
    ap.add_argument("--out", default="preprocess_out/phase0", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    face_a = detect_face(args.makeup_image)
    uv_a, vis_a = unwrap(face_a.image, face_a.landmarks, UV_SIZE)
    uv_a, vis_a = mirror_fill(uv_a, vis_a)

    layer, skin_lab = derive_layer(uv_a, vis_a)
    store = LookStore()
    store.save(
        args.look,
        layer,
        meta={
            "source_with": str(args.makeup_image),
            "blend": "lab-delta",
            "notes": json.dumps({"skin_lab_mean": [round(float(v), 2) for v in skin_lab]}),
        },
    )

    face_b = detect_face(args.target_image)
    result = apply_layer(face_b.image, face_b.landmarks, layer, skin_lab)

    cv2.imwrite(str(out / "uv_makeup_source.png"), uv_a)
    cv2.imwrite(str(out / "layer_preview.png"), _checker_preview(layer))
    cv2.imwrite(str(out / "result.png"), result)
    side = np.hstack([face_b.image, result])
    cv2.imwrite(str(out / "side_by_side.png"), side)
    print(f"Saved look '{args.look}' to {store.root / args.look}")
    print(f"Wrote uv_makeup_source / layer_preview / result / side_by_side to {out}/")


if __name__ == "__main__":
    _main()
