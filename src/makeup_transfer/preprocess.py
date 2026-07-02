"""Preprocessing: detect a face and produce landmarks + region masks.

This is the front of the pipeline shared by both extraction and application. It is intentionally
dependency-light (MediaPipe FaceLandmarker + OpenCV) so it runs without the heavy 3D / GAN repos.

The 3D fit -> UV unwrap step (3DDFA_V2 / DECA) is added in a later module; this file provides the
2D landmarks and per-region masks those steps and the blending logic rely on.

Uses the MediaPipe **Tasks** API (the legacy ``mp.solutions`` API is gone in recent wheels). The
model bundle ``face_landmarker.task`` is expected under ``weights/`` (downloaded during setup).

Quick check::

    python -m makeup_transfer.preprocess path/to/face.jpg --out out_dir
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Default location of the FaceLandmarker model bundle (repo_root/weights/face_landmarker.task).
_MODEL_PATH = Path(__file__).resolve().parents[2] / "weights" / "face_landmarker.task"

# MediaPipe Face Mesh canonical landmark groups (subset we need for makeup regions).
# Indices are stable across MediaPipe versions for the 468-point mesh.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
    0, 37, 39, 40, 185,
]
# Inner lip ring = the mouth opening (teeth/tongue when open). Excluded from makeup mattes.
LIPS_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 312, 13,
    82, 81, 80, 191,
]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

REGION_GROUPS = {
    "face": FACE_OVAL,
    "lips": LIPS_OUTER,
    "mouth": LIPS_INNER,
    "left_eye": LEFT_EYE,
    "right_eye": RIGHT_EYE,
    "left_brow": LEFT_BROW,
    "right_brow": RIGHT_BROW,
}


@dataclass
class FaceData:
    """Result of preprocessing a single face image."""

    image: np.ndarray            # original BGR image (H, W, 3) uint8
    landmarks: np.ndarray        # (468, 2) float32 pixel coords
    masks: dict[str, np.ndarray] # region name -> (H, W) uint8 mask {0,255}

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


def _landmarks_to_pixels(landmarks, w: int, h: int) -> np.ndarray:
    # FaceLandmarker returns the 468 mesh points first (then 10 iris points with refine);
    # we keep the first 468 so the canonical REGION_GROUPS indices stay valid.
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks[:468]], dtype=np.float32)
    return pts


def _polygon_mask(points: np.ndarray, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def detect_face(image_path: str | Path, model_path: str | Path | None = None) -> FaceData:
    """Run MediaPipe FaceLandmarker on one image and build region masks.

    Raises ``ValueError`` if no face is found, ``FileNotFoundError`` if image/model missing.
    """
    # Imported lazily so the module can be imported without mediapipe present.
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model = Path(model_path) if model_path is not None else _MODEL_PATH
    if not model.exists():
        raise FileNotFoundError(
            f"FaceLandmarker model not found at {model}. Download it with:\n"
            "  curl -L -o weights/face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )

    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError(f"No face detected in {path}")

    landmarks = _landmarks_to_pixels(result.face_landmarks[0], w, h)

    masks: dict[str, np.ndarray] = {}
    for name, idxs in REGION_GROUPS.items():
        masks[name] = _polygon_mask(landmarks[idxs], h, w)

    return FaceData(image=image, landmarks=landmarks, masks=masks)


def overlay_masks(face: FaceData) -> np.ndarray:
    """Return a debug visualization: image with region masks tinted, for eyeballing."""
    vis = face.image.copy()
    colors = {
        "lips": (60, 60, 220), "left_eye": (220, 160, 60), "right_eye": (220, 160, 60),
        "left_brow": (60, 220, 60), "right_brow": (60, 220, 60), "face": (200, 200, 60),
    }
    for name, mask in face.masks.items():
        if name == "face":
            continue
        color = np.array(colors.get(name, (0, 255, 255)), dtype=np.uint8)
        vis[mask > 0] = (0.5 * vis[mask > 0] + 0.5 * color).astype(np.uint8)
    for (x, y) in face.landmarks.astype(int):
        cv2.circle(vis, (x, y), 1, (255, 255, 255), -1)
    return vis


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Preprocess a face: landmarks + region masks.")
    ap.add_argument("image", help="path to a face image")
    ap.add_argument("--out", default="preprocess_out", help="output directory")
    args = ap.parse_args()

    face = detect_face(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "landmarks_overlay.png"), overlay_masks(face))
    for name, mask in face.masks.items():
        cv2.imwrite(str(out / f"mask_{name}.png"), mask)
    print(f"Detected {len(face.landmarks)} landmarks; "
          f"wrote overlay + {len(face.masks)} masks to {out}/")


if __name__ == "__main__":
    _main()
