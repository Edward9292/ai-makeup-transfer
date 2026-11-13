"""Occlusion masking: which face-chart pixels are actually *skin* (``occlusion.py``).

The landmark mesh has no idea what is in front of the face. Anything the face oval covers —
a fringe of hair, a hand, glasses, a scarf — is unwrapped as if it were skin, which breaks the
pipeline at both ends:

- **extraction** — hair falling across the forehead/brow differs wildly between the with-makeup
  and bare shots, so the Lab difference reads it as makeup and bakes hair into the layer;
- **application** — the makeup delta is projected onto the whole chart footprint, so brow or
  eyeshadow color is painted *on top of* whatever hair covers that spot in the target.

Both are fixed by one signal: a per-pixel confidence that the pixel is exposed facial skin.
It comes from MediaPipe's **selfie multiclass segmenter** (``selfie_multiclass_256x256.tflite``
under ``weights/``), which labels background / hair / body-skin / face-skin / clothes /
accessories. Brows, eyes and lips are labelled face-skin by that model, so makeup regions are
kept while hair over them is not.

The model is optional: if the weight file is missing (or MediaPipe cannot load it) every
function here returns ``None`` and callers fall back to the previous no-occlusion behavior,
so the pipeline still runs on a machine that never downloaded it.

Quick check::

    python -m makeup_transfer.occlusion path/to/face.jpg --out preprocess_out
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Default location of the segmenter bundle (repo_root/weights/selfie_multiclass_256x256.tflite).
MODEL_PATH = Path(__file__).resolve().parents[2] / "weights" / "selfie_multiclass_256x256.tflite"

DOWNLOAD_HINT = (
    "  curl -L -o weights/selfie_multiclass_256x256.tflite "
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)

# Class order of the selfie multiclass model.
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, ACCESSORIES = range(6)
# Skin classes: face-skin is the one we care about, body-skin keeps the neck/jaw edge of the
# face chart from being treated as an occluder on a tilted head.
_SKIN_CLASSES = (FACE_SKIN, BODY_SKIN)

# Confidence ramp. Below LO the pixel is treated as fully occluded, above HI as fully exposed;
# the model's 256x256 output is soft at boundaries, so the ramp feathers hair edges instead of
# producing a hard, aliased cut-out.
CONF_LO = 0.25
CONF_HI = 0.60

# Occluders are dilated by this fraction of the image's smaller side before the ramp: the
# segmenter runs at 256x256 and consistently under-covers wispy hair edges by a pixel or two.
GROW_FRAC = 0.008


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def available(model_path: str | Path | None = None) -> bool:
    """Is the segmenter weight present? (Cheap check, no model load.)"""
    return (Path(model_path) if model_path is not None else MODEL_PATH).exists()


def skin_visibility(
    image_bgr: np.ndarray,
    model_path: str | Path | None = None,
) -> np.ndarray | None:
    """Per-pixel confidence in [0, 1] that a pixel shows exposed facial skin.

    ``1`` = skin (including brows, eyes and lips), ``0`` = occluder (hair, hands, glasses,
    clothing) or background. Returns ``None`` — meaning "no occlusion information, carry on" —
    when the model file is absent or MediaPipe fails to run it.
    """
    model = Path(model_path) if model_path is not None else MODEL_PATH
    if not model.exists():
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with mp_vision.ImageSegmenter.create_from_options(options) as segmenter:
            result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            # numpy_view() is a view onto memory the segmenter owns; copy before it closes.
            skin = np.sum(
                [np.squeeze(result.confidence_masks[c].numpy_view()) for c in _SKIN_CLASSES],
                axis=0,
            ).astype(np.float32)
    except Exception:  # noqa: BLE001 — occlusion masking is strictly an enhancement
        return None

    h, w = image_bgr.shape[:2]
    if skin.shape != (h, w):
        skin = cv2.resize(skin, (w, h), interpolation=cv2.INTER_LINEAR)

    # Erode the skin (= dilate occluders) so thin hair strands the 256px model clips are covered.
    px = max(int(round(GROW_FRAC * min(h, w))), 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    skin = cv2.erode(skin, k)
    return _smoothstep(skin, CONF_LO, CONF_HI)


def skin_visibility_uv(face, uv_size: int) -> np.ndarray | None:
    """A ``FaceData``'s skin mask unwrapped into canonical UV space (or ``None``, as above).

    Reads ``face.skin_visibility``, so the segmenter runs at most once per detected face.
    """
    from .uv_unwrap import unwrap

    skin = face.skin_visibility
    if skin is None:
        return None
    # unwrap() remaps uint8 image data; round-trip through 8-bit is plenty for a soft mask.
    uv_skin, _ = unwrap((skin * 255).astype(np.uint8), face.landmarks, uv_size)
    return uv_skin.astype(np.float32) / 255.0


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Visualize the skin/occluder mask for a face photo.")
    ap.add_argument("image", help="path to a face image")
    ap.add_argument("--out", default="preprocess_out", help="output directory")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    skin = skin_visibility(image)
    if skin is None:
        raise SystemExit(f"Segmenter weight not found at {MODEL_PATH}. Download it with:\n"
                         f"{DOWNLOAD_HINT}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    cv2.imwrite(str(out / f"skin_{stem}.png"), (skin * 255).astype(np.uint8))
    tint = np.zeros_like(image)
    tint[..., 2] = 255  # occluders in red
    overlay = (image * skin[..., None] + 0.5 * (1.0 - skin[..., None]) * (image + tint))
    cv2.imwrite(str(out / f"skin_{stem}_overlay.png"), np.clip(overlay, 0, 255).astype(np.uint8))
    print(f"Skin coverage {skin.mean():.1%}; wrote mask + overlay to {out}/")


if __name__ == "__main__":
    _main()
