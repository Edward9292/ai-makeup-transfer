"""Phase 2: apply a saved makeup look to a target photo (``apply.py``).

**Engine looks** (``reference.png``, the current default) are rendered by ``engine.py``: the
transfer network reads the makeup off the stored reference face and paints it onto the target,
composited at full resolution. ``apply_look`` dispatches there first; everything below is the
**legacy UV-layer path**, kept so older looks (``layer.png``) still apply.

Legacy pipeline: load look (UV RGBA layer + ``meta.json``) -> detect target face -> unwrap target into
the same canonical UV space -> blend the layer over the target texture with **per-region blend
modes** -> re-project to the photo and composite.

The blend is *skin-tone-aware*: every effect is computed relative to the **source's bare skin**
(``skin_lab_mean``, saved with the look), so the target keeps its own luminance base and skin
texture and the same look adapts across skin tones instead of pasting the source's colors on.

Per-region modes (BUILD_PLAN Phase 2 — one global blend is physically wrong):

- ``multiply``  (lips, brows) — absorptive pigment. Target L is darkened *multiplicatively* by
  the layer's transmittance relative to source skin (Beer-Lambert-ish; alpha acts as optical
  depth), so lip texture/highlights show through. Brightening (gloss shine) is added on top,
  damped on already-bright skin.
- ``softlight`` (eye shadow, blush/powder on skin) — scattering. W3C soft-light on L, which
  self-damps near black/white; gentler than multiply.
- ``additive``  (glitter/highlighter) — specular. Adds only the brightening component.

Which region gets which mode + strength lives in the look's ``meta.json`` (``blend_params``),
so it is editable per look; ``--strength region=x`` overrides at apply time.

The final composite is gated by the target's skin mask (``occlusion.py``), so makeup lands on
skin only — hair falling over a brow or eye stays hair instead of being tinted through.

Run::

    python -m makeup_transfer.apply ffhq-002386-03 data/ffhq_makeup/003417_bare.jpg \
        --out preprocess_out/phase2 --strength lips=1.0 --strength eyes=0.8
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .preprocess import detect_face
from .storage import LookMeta, LookStore
from .uv_unwrap import project, region_masks_uv, unwrap

UV_SIZE = 512

# Default per-region recipe; extract.py copies this into each new look's meta.json.
DEFAULT_REGION_PARAMS: dict[str, dict] = {
    "lips":  {"mode": "multiply",  "strength": 1.0},
    "brows": {"mode": "multiply",  "strength": 0.9},
    "eyes":  {"mode": "softlight", "strength": 1.0},
    "skin":  {"mode": "softlight", "strength": 1.0},
}

# How far multiplicative darkening may go (transmittance floor): stops near-black lipstick on
# already-dark skin from crushing to 0 and losing all texture.
_MIN_TRANSMITTANCE = 0.15


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


def region_weights_uv(uv_size: int = UV_SIZE) -> dict[str, np.ndarray]:
    """Soft, per-pixel weights for the four blend regions in canonical UV space.

    Feathered and normalized so the weights sum to 1 inside the face chart — every makeup
    pixel gets exactly one convex combination of modes, with smooth hand-offs at borders.
    """
    s = uv_size
    px = max(s // 100, 1)
    masks = region_masks_uv(s)

    eyes = masks["left_eye"] | masks["right_eye"]
    brows = _dilate(masks["left_brow"] | masks["right_brow"], px)
    lips = _dilate(masks["lips"], px)
    # Eyeshadow/liner band: generous halo around the eye openings, brows excluded.
    eye_band = _dilate(eyes, 8 * px) & ~brows & ~lips
    face = masks["face"]
    skin = face & ~lips & ~brows & ~eye_band

    hard = {"lips": lips, "brows": brows, "eyes": eye_band, "skin": skin}
    soft = {
        name: cv2.GaussianBlur((m & face).astype(np.float32) / 255.0, (0, 0), 2.0 * px)
        for name, m in hard.items()
    }
    total = np.maximum(sum(soft.values()), 1e-6)
    return {name: m / total for name, m in soft.items()}


def _softlight(t: np.ndarray, l: np.ndarray) -> np.ndarray:
    """W3C soft-light, single channel, both in [0, 1]."""
    d = np.where(t <= 0.25, ((16.0 * t - 12.0) * t + 4.0) * t, np.sqrt(np.maximum(t, 0.0)))
    dark = t - (1.0 - 2.0 * l) * t * (1.0 - t)
    light = t + (2.0 * l - 1.0) * (d - t)
    return np.where(l <= 0.5, dark, light)


def _blend_mode_lab(
    mode: str,
    lab_t: np.ndarray,
    lab_l: np.ndarray,
    skin_lab: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Blend layer over target in Lab with per-pixel weight ``w`` (already alpha*strength)."""
    L_t, ab_t = lab_t[..., 0], lab_t[..., 1:]
    L_l, ab_l = lab_l[..., 0], lab_l[..., 1:]
    L_s = max(float(skin_lab[0]), 1.0)

    out = lab_t.copy()
    if mode == "multiply":
        ratio = np.clip(L_l / L_s, _MIN_TRANSMITTANCE, 1.0)
        L_out = L_t * np.power(ratio, w)                       # alpha as optical depth
        gloss = np.clip(L_l - L_s, 0.0, None) * (1.0 - L_t / 100.0)
        L_out = L_out + w * gloss
        ab_w = w
    elif mode == "softlight":
        l_eff = np.clip(0.5 + (L_l - L_s) / 100.0, 0.0, 1.0)
        L_out = L_t + w * (100.0 * _softlight(L_t / 100.0, l_eff) - L_t)
        ab_w = w
    elif mode == "additive":
        L_out = L_t + w * np.clip(L_l - L_s, 0.0, None)
        ab_w = 0.5 * w  # specular is mostly white; pull chroma only lightly
    else:
        raise ValueError(f"Unknown blend mode '{mode}'")

    out[..., 0] = np.clip(L_out, 0.0, 100.0)
    out[..., 1:] = (1.0 - ab_w[..., None]) * ab_t + ab_w[..., None] * ab_l
    return out


def _resolve_params(meta: LookMeta, strengths: dict[str, float] | None):
    """Merge meta blend_params with defaults + CLI overrides -> (regions, skin_lab_mean)."""
    bp = meta.blend_params or {}
    regions = {name: dict(p) for name, p in DEFAULT_REGION_PARAMS.items()}
    for name, p in bp.get("regions", {}).items():
        regions.setdefault(name, {}).update(p)
    for name, s in (strengths or {}).items():
        if name not in regions:
            raise ValueError(f"Unknown region '{name}' (have: {sorted(regions)})")
        regions[name]["strength"] = float(s)

    skin_lab = bp.get("skin_lab_mean")
    if skin_lab is None and meta.notes:  # older looks stashed it in notes JSON
        try:
            skin_lab = json.loads(meta.notes).get("skin_lab_mean")
        except (json.JSONDecodeError, AttributeError):
            skin_lab = None
    if skin_lab is None:
        raise ValueError(
            f"Look '{meta.name}' has no skin_lab_mean in its metadata; re-extract it "
            "(apply needs the source's bare-skin color to adapt the look to the target)."
        )
    return regions, np.asarray(skin_lab, dtype=np.float32)


def blend_uv(
    uv_target: np.ndarray,
    layer_rgba: np.ndarray,
    skin_lab_mean: np.ndarray,
    regions: dict[str, dict] | None = None,
) -> np.ndarray:
    """Blend an RGBA makeup layer over a target UV texture (both same size). Returns BGR."""
    regions = regions or DEFAULT_REGION_PARAMS
    s = uv_target.shape[0]
    alpha = layer_rgba[..., 3].astype(np.float32) / 255.0
    layer_bgr = cv2.cvtColor(layer_rgba[..., :3], cv2.COLOR_RGB2BGR)

    lab_t = _rgb_lab(uv_target)
    lab_l = _rgb_lab(layer_bgr)
    weights = region_weights_uv(s)

    out = np.zeros_like(lab_t)
    covered = np.zeros((s, s), dtype=np.float32)
    for name, params in regions.items():
        rw = weights.get(name)
        if rw is None:
            continue
        w = alpha * float(params.get("strength", 1.0)) * rw
        out += rw[..., None] * _blend_mode_lab(
            params.get("mode", "softlight"), lab_t, lab_l, skin_lab_mean, w
        )
        covered += rw
    # Outside the union of region weights (e.g. outside the face chart) keep the target as-is.
    out += np.clip(1.0 - covered, 0.0, 1.0)[..., None] * lab_t
    return _lab_bgr(out)


def apply_look(
    look_name: str,
    target_image: str | Path,
    *,
    store: LookStore | None = None,
    strengths: dict[str, float] | None = None,
    uv_size: int = UV_SIZE,
    composite: str = "delta",
    occlusion: bool = True,
    complexion: float | None = None,
):
    """Apply a saved look to a target photo. Returns ``(result_bgr, debug)``.

    Engine looks (``store.has_reference``) go through ``engine.transfer_makeup``; ``complexion``
    (0..1, how much of the reference's skin tone to take) applies to them only and defaults to
    the look's own setting. The parameters below describe the legacy UV-layer path.

    ``composite="delta"`` (default) harmonizes against the double-resample blur of the
    image -> UV -> image round trip: only the makeup *change* (blended - target UV texture,
    both through the same round trip, so resampling softness cancels) is projected and added
    onto the untouched original photo, preserving the target's full-resolution skin grain
    under the makeup. ``composite="plain"`` is the naive alpha-over for comparison.

    ``occlusion=True`` (default) keeps the makeup off anything covering the target's face —
    hair, a hand, glasses — via the segmenter mask (see ``occlusion.py``); set it to ``False``
    for the previous, occlusion-blind composite.
    """
    store = store or LookStore()
    meta = store.load_meta(look_name)
    if (reference := store.load_reference(look_name)) is not None:
        from .engine import DEFAULT_COMPLEXION, transfer_makeup

        bp = meta.blend_params or {}
        region_strengths = {k: float(v.get("strength", 1.0))
                            for k, v in bp.get("regions", {}).items()}
        region_strengths.update(strengths or {})
        if complexion is None:
            complexion = float(bp.get("complexion", DEFAULT_COMPLEXION))
        res = transfer_makeup(target_image, reference, strengths=region_strengths,
                              complexion=complexion, occlusion=occlusion)
        debug = {"target": res.target, "net_in": res.net_in, "net_out": res.net_out,
                 "reference": res.reference_crop, "mask": res.mask, "crop": res.crop}
        return res.image, debug

    layer = store.load_layer(look_name)
    regions, skin_lab = _resolve_params(meta, strengths)

    if layer.shape[0] != uv_size:
        layer = cv2.resize(layer, (uv_size, uv_size), interpolation=cv2.INTER_LINEAR)

    face = detect_face(target_image)
    uv_t, _ = unwrap(face.image, face.landmarks, uv_size)
    blended_uv = blend_uv(uv_t, layer, skin_lab, regions)

    h, w = face.image.shape[:2]
    alpha_img, _ = project(layer[..., 3], face.landmarks, h, w)
    a = (alpha_img.astype(np.float32) / 255.0)[..., None]
    # The mesh spans the face oval regardless of what is actually in front of it, so hair
    # falling over a brow or eye would get painted with that region's color. Gate the
    # composite by the target's own skin mask, leaving occluders untouched.
    if occlusion and (skin := face.skin_visibility) is not None:
        a = a * skin[..., None]
    blended_img, _ = project(blended_uv, face.landmarks, h, w)
    if composite == "delta":
        base_img, _ = project(uv_t, face.landmarks, h, w)
        delta = blended_img.astype(np.float32) - base_img.astype(np.float32)
        result = np.clip(face.image.astype(np.float32) + a * delta, 0, 255).astype(np.uint8)
    elif composite == "plain":
        result = ((1.0 - a) * face.image.astype(np.float32)
                  + a * blended_img.astype(np.float32)).astype(np.uint8)
    else:
        raise ValueError(f"Unknown composite '{composite}' (delta|plain)")

    debug = {"uv_target": uv_t, "uv_blended": blended_uv, "alpha_img": alpha_img,
             "target": face.image, "alpha_gated": (a[..., 0] * 255).astype(np.uint8)}
    return result, debug


def _parse_strengths(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items or []:
        name, _, val = item.partition("=")
        if not val:
            raise SystemExit(f"--strength expects region=value, got '{item}'")
        out[name.strip()] = float(val)
    return out


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Apply a saved makeup look to a target photo.")
    ap.add_argument("look", help="look name (folder under looks/)")
    ap.add_argument("target", help="target photo (face without makeup)")
    ap.add_argument("--out", default="preprocess_out/phase2", help="output directory")
    ap.add_argument("--size", type=int, default=UV_SIZE, help="UV texture size")
    ap.add_argument("--strength", action="append", metavar="REGION=X",
                    help="override a region strength, e.g. --strength lips=0.8 (repeatable)")
    ap.add_argument("--composite", default="delta", choices=["delta", "plain"],
                    help="delta = grain-preserving harmonized composite (default)")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="don't mask out hair/hands covering the target's face")
    ap.add_argument("--complexion", type=float, default=None,
                    help="engine looks: 0..1 share of the reference's skin tone to take")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result, debug = apply_look(
        args.look, args.target, strengths=_parse_strengths(args.strength),
        uv_size=args.size, composite=args.composite, occlusion=not args.no_occlusion,
        complexion=args.complexion,
    )

    stem = f"{args.look}_on_{Path(args.target).stem}"
    cv2.imwrite(str(out / f"{stem}.png"), result)
    cv2.imwrite(str(out / f"{stem}_side.png"), np.hstack([debug["target"], result]))
    print(f"Wrote {out / f'{stem}.png'} (+ _side.png)")


if __name__ == "__main__":
    _main()
