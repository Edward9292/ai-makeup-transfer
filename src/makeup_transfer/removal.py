"""Phase 2.5: single-image bare-face estimation (makeup removal -> pseudo-pair).

Wraps two vendored, inference-only third-party repos (nothing is compiled):

- ``third_party/SSAT/quick_start`` -- SSAT (AAAI 2022) makeup transfer/removal GAN.
  Weights ship in that repo. Removal is style transfer of "no makeup", so it needs a
  bare *reference* face; any reasonably frontal bare photo works (default below).
- ``third_party/face-parsing.PyTorch`` -- BiSeNet face parsing, the parser SSAT was
  trained with (18 semantic classes; label 18 "hat" is folded into background).

The pseudo-bare output is the *input photograph* with the face crop regenerated at
SSAT's 256x256 training size and pasted back, so the (makeup, pseudo-bare) pair is
pixel-aligned by construction and flows through ``extract.py`` like a real pair.
The 256px bottleneck only degrades the alpha matte, not the layer color (which
comes from the full-res with-makeup photo).

CUDA is required: SSAT's ``init_net`` asserts it.

Run::

    PYTHONPATH=src python -m makeup_transfer.removal data/ffhq_makeup/002386_makeup_03.jpg \
        --out preprocess_out/phase25/pseudo_bare.png
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

from .preprocess import detect_face

_ROOT = Path(__file__).resolve().parents[2]
SSAT_DIR = _ROOT / "third_party" / "SSAT" / "quick_start"
PARSING_DIR = _ROOT / "third_party" / "face-parsing.PyTorch"
SSAT_WEIGHTS = SSAT_DIR / "weights" / "SSAT.pth"
PARSING_WEIGHTS = PARSING_DIR / "res" / "cp" / "79999_iter.pth"
DEFAULT_REFERENCE = _ROOT / "data" / "ffhq_makeup" / "003125_bare.jpg"

SSAT_SIZE = 256      # SSAT training resolution
PARSE_SIZE = 512     # BiSeNet eval resolution
SEMANTIC_DIM = 18    # SSAT one-hot classes (face-parsing labels 0-17; 18 "hat" -> 0)
CROP_EXPAND = 1.6    # square crop half-size = this * max landmark extent / 2


def _load_module(alias: str, path: Path):
    """Import a third-party file under a controlled name (both repos have model.py).

    Some aliases (e.g. "networks", "resnet") are forced: the vendored repos use flat
    imports (``import networks``, ``from resnet import ...``), so those modules must
    occupy bare top-level names in sys.modules -- they can't be namespaced without
    editing third_party code.
    """
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _crop_box(landmarks: np.ndarray, shape: tuple) -> tuple[int, int, int, int]:
    """Square face crop around the landmarks, clamped to the frame."""
    h, w = shape[:2]
    lo, hi = landmarks.min(axis=0), landmarks.max(axis=0)
    cx, cy = (lo + hi) / 2.0
    half = float((hi - lo).max()) / 2.0 * CROP_EXPAND
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    return x0, y0, x1, y1


class BareFaceEstimator:
    """SSAT makeup removal behind a small, swappable interface (see design doc)."""

    def __init__(self, default_reference: str | Path = DEFAULT_REFERENCE,
                 device_index: int = 0):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("SSAT removal requires CUDA (its init_net asserts it).")
        for pth, hint in ((SSAT_WEIGHTS, "git clone https://github.com/Snowfallingplum/SSAT third_party/SSAT"),
                          (PARSING_WEIGHTS, "see BUILD_PLAN phase 2.5: gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812")):
            if not pth.exists():
                raise FileNotFoundError(f"Missing weights: {pth}\n  -> {hint}")
        self._torch = torch
        self.device = torch.device(f"cuda:{device_index}")
        self.default_reference = Path(default_reference)
        self._ref_cache: dict[Path, tuple] = {}
        self._load_nets(device_index)

    # --- model loading -----------------------------------------------------
    def _load_nets(self, gpu: int) -> None:
        torch = self._torch
        _load_module("networks", SSAT_DIR / "networks.py")   # ssat model.py does `import networks`
        ssat_model = _load_module("ssat_qs_model", SSAT_DIR / "model.py")
        opts = argparse.Namespace(gpu=gpu, input_dim=3, output_dim=3,
                                  semantic_dim=SEMANTIC_DIM, batch_size=1, lr=2e-4)
        self.gan = ssat_model.MakeupGAN(opts)
        ck = torch.load(SSAT_WEIGHTS, map_location=self.device, weights_only=False)
        self.gan.enc_content.load_state_dict(ck["enc_c"])
        self.gan.enc_makeup.load_state_dict(ck["enc_a"])
        self.gan.enc_semantic.load_state_dict(ck["enc_s"])
        self.gan.transformer.load_state_dict(ck["enc_trans"])
        self.gan.gen.load_state_dict(ck["gen"])
        self.gan.eval()

        _load_module("resnet", PARSING_DIR / "resnet.py")    # bisenet model.py does `from resnet import ...`
        bisenet_mod = _load_module("bisenet_model", PARSING_DIR / "model.py")
        self.bisenet = bisenet_mod.BiSeNet(n_classes=19)     # downloads resnet18 backbone once
        self.bisenet.load_state_dict(
            torch.load(PARSING_WEIGHTS, map_location=self.device, weights_only=False))
        self.bisenet.to(self.device).eval()

    # --- per-image preparation ----------------------------------------------
    def _parse(self, crop_bgr: np.ndarray) -> np.ndarray:
        """BiSeNet labels for a face crop, returned at SSAT_SIZE. Labels 0-17."""
        torch = self._torch
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (PARSE_SIZE, PARSE_SIZE)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(self.device) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        with torch.no_grad():
            out = self.bisenet((x - mean) / std)[0]
        labels = out.argmax(1)[0].byte().cpu().numpy()
        labels[labels >= SEMANTIC_DIM] = 0  # "hat" (18) has no SSAT channel
        return cv2.resize(labels, (SSAT_SIZE, SSAT_SIZE), interpolation=cv2.INTER_NEAREST)

    def _tensors(self, image_path: str | Path):
        """photo -> (FaceData, crop box, image tensor, one-hot parse tensor)."""
        torch = self._torch
        face = detect_face(image_path)
        box = _crop_box(face.landmarks, face.image.shape)
        x0, y0, x1, y1 = box
        crop = cv2.resize(face.image[y0:y1, x0:x1], (SSAT_SIZE, SSAT_SIZE),
                          interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(self.device)
        labels = self._parse(crop)
        one_hot = np.zeros((SEMANTIC_DIM, SSAT_SIZE, SSAT_SIZE), dtype=np.float32)
        for i in range(SEMANTIC_DIM):
            one_hot[i][labels == i] = 1.0
        parse_t = torch.from_numpy(one_hot)[None].to(self.device)
        return face, box, img_t, parse_t

    def _reference_tensors(self, reference: Path):
        if reference not in self._ref_cache:
            _, _, img_t, parse_t = self._tensors(reference)
            self._ref_cache[reference] = (img_t, parse_t)
        return self._ref_cache[reference]

    # --- public API ----------------------------------------------------------
    def estimate_bare(self, image_path: str | Path,
                      reference: str | Path | None = None) -> np.ndarray:
        """Synthesize a bare version of a with-makeup photo (full frame, BGR uint8)."""
        torch = self._torch
        ref_img, ref_parse = self._reference_tensors(Path(reference or self.default_reference))
        face, box, img_t, parse_t = self._tensors(image_path)
        data = {"non_makeup": ref_img, "makeup": img_t,
                "non_makeup_parse": ref_parse, "makeup_parse": parse_t}
        with torch.no_grad():
            self.gan.test_pair(data)          # populates gan.z_removal
            removal_arr = self.gan.z_removal[0].cpu().numpy()
        out = ((removal_arr.transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        x0, y0, x1, y1 = box
        result = face.image.copy()
        result[y0:y1, x0:x1] = cv2.resize(out_bgr, (x1 - x0, y1 - y0),
                                          interpolation=cv2.INTER_CUBIC)
        return result


_estimator: BareFaceEstimator | None = None


def estimate_bare(image_path: str | Path,
                  reference: str | Path | None = None) -> np.ndarray:
    """Module-level convenience with a lazily created, cached estimator."""
    global _estimator
    if _estimator is None:
        _estimator = BareFaceEstimator()
    return _estimator.estimate_bare(image_path, reference=reference)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize a bare face from a makeup photo (SSAT removal).")
    ap.add_argument("image", help="with-makeup photo")
    ap.add_argument("--ref", default=None, help="bare reference photo (style source)")
    ap.add_argument("--out", default="preprocess_out/phase25/pseudo_bare.png")
    args = ap.parse_args()
    result = estimate_bare(args.image, reference=args.ref)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), result)
    print(f"Wrote {out}")


if __name__ == "__main__":
    _main()
