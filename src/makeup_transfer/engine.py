"""Reference-based makeup transfer engine (``engine.py``).

Replaces the pair-difference extractor + blend-mode compositor as the way makeup gets onto a
face. Input is simply **a target photo and a reference photo of someone wearing the makeup**;
no same-person pair, no UV layer. The network (CSD-MT, CVPR 2024 — content/style decoupling,
unsupervised, pose/expression robust, small enough for CPU) renders the reference's makeup
onto a 256 px crop of the target face, and the crop is composited back onto the full-resolution
photo as a *delta* so the target keeps its own skin texture, lighting and resolution.

Why not just return the network output: the generator works at 256 px and, like any GAN
reconstruction, softens skin grain and drifts a little outside the makeup regions. Adding only
``(output - input)`` to the original, gated by a feathered face-parsing mask and the segmenter's
skin-visibility map (hair / hands / glasses stay untouched), keeps the makeup and discards the
rest. Per-region strengths (lips / eyes / brows / skin) scale that delta, so the same sliders
the old pipeline exposed keep working.

Backends are pluggable (``Engine.transfer_crop``): CSD-MT is the default CPU engine; a diffusion
renderer (Stable-Makeup / FLUX-Makeup) can be dropped in behind the same crop -> delta path.

Run::

    python -m makeup_transfer.engine data/ffhq_makeup/003417_bare.jpg \
        data/ffhq_makeup/002386_makeup_03.jpg --out preprocess_out/engine
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .preprocess import FaceData, detect_face, detect_face_bgr

_WEIGHTS = Path(__file__).resolve().parents[2] / "weights"
CSDMT_WEIGHTS = _WEIGHTS / "csd_mt.pth"
BISENET_WEIGHTS = _WEIGHTS / "bisenet_79999_iter.pth"

NET_SIZE = 256          # CSD-MT training resolution
PARSE_SIZE = 512        # BiSeNet input resolution
REFERENCE_SIZE = 512    # stored reference crop size (looks/<name>/reference.png)

# Crop geometry copied from CSD-MT's dlib-based crop (margins relative to the face box) so the
# network sees faces framed the way it was trained, but driven by MediaPipe landmarks instead
# of dlib (one fewer native dependency; the landmarks are already computed for occlusion).
_UP_RATIO = 0.2 / 0.85
_DOWN_RATIO = 0.15 / 0.85
_WIDTH_RATIO = 0.2 / 0.85

# BiSeNet (CelebAMask-HQ, 19 classes) labels.
_BG, _SKIN, _L_BROW, _R_BROW, _L_EYE, _R_EYE, _GLASSES, _L_EAR, _R_EAR, _EARRING = range(10)
_NOSE, _MOUTH, _U_LIP, _L_LIP, _NECK, _NECKLACE, _CLOTH, _HAIR, _HAT = range(10, 19)

DEFAULT_STRENGTHS: dict[str, float] = {"lips": 1.0, "eyes": 1.0, "brows": 1.0, "skin": 1.0}


def available() -> bool:
    """Are the engine weights present (cheap check, no import of torch)?"""
    return CSDMT_WEIGHTS.exists() and BISENET_WEIGHTS.exists()


def missing_weights_message() -> str:
    return (
        "Makeup engine weights not found. Fetch them with:\n"
        "  python -m makeup_transfer.fetch_weights\n"
        f"(expects {CSDMT_WEIGHTS.name} and {BISENET_WEIGHTS.name} under {_WEIGHTS}/)"
    )


# --- face crop ---------------------------------------------------------------------------------
@dataclass
class Crop:
    """A square crop ``image[y0:y0+size, x0:x0+size]`` (clamped to the image)."""

    x0: int
    y0: int
    size: int

    def slice(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y0 + self.size), slice(self.x0, self.x0 + self.size)


def face_box(face: FaceData) -> tuple[int, int, int, int]:
    """dlib-style face rectangle (left, top, right, bottom) from the landmarks: brow line to
    chin, cheek to cheek. dlib's detector box sits roughly there, which is what the crop margins
    were tuned against."""
    pts = face.landmarks
    from .preprocess import FACE_OVAL, LEFT_BROW, RIGHT_BROW

    oval = pts[FACE_OVAL]
    brows = pts[LEFT_BROW + RIGHT_BROW]
    left, right = float(oval[:, 0].min()), float(oval[:, 0].max())
    top = float(brows[:, 1].min())
    bottom = float(oval[:, 1].max())
    return int(left), int(top), int(right), int(bottom)


# "portrait" framing (diffusion engine): FFHQ-like head-and-shoulders crop — the square's side
# is this many face-widths, centred a little above the face-box centre.
_PORTRAIT_SCALE = 1.8
_PORTRAIT_UP = 0.10


def face_crop(face: FaceData, framing: str = "tight") -> Crop:
    """Square crop around the face, clamped to the photo.

    ``"tight"`` uses CSD-MT's margins (face fills the frame); ``"portrait"`` is the looser
    FFHQ-style framing Stable Diffusion face models are trained on.
    """
    h, w = face.image.shape[:2]
    left, top, right, bottom = face_box(face)
    fh = bottom - top
    if framing == "portrait":
        size = int(min(_PORTRAIT_SCALE * (right - left), h, w))
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0 - _PORTRAIT_UP * fh
        sx = int(round(min(max(cx - size / 2.0, 0), w - size)))
        sy = int(round(min(max(cy - size / 2.0, 0), h - size)))
        return Crop(sx, sy, size)
    x0 = max(0, int(left - _WIDTH_RATIO * (right - left)))
    x1 = min(w, int(right + _WIDTH_RATIO * (right - left)))
    y0 = max(0, int(top - _UP_RATIO * fh))
    y1 = min(h, int(bottom + _DOWN_RATIO * fh))
    cw, ch = x1 - x0, y1 - y0
    size = min(cw, ch)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx = int(round(min(max(cx - size / 2.0, 0), w - size)))
    sy = int(round(min(max(cy - size / 2.0, 0), h - size)))
    return Crop(sx, sy, size)


def crop_image(image: np.ndarray, crop: Crop, size: int) -> np.ndarray:
    ys, xs = crop.slice()
    interp = cv2.INTER_AREA if crop.size > size else cv2.INTER_CUBIC
    return cv2.resize(image[ys, xs], (size, size), interpolation=interp)


# --- the network -------------------------------------------------------------------------------
def _device():
    import torch

    want = os.environ.get("MAKEUP_DEVICE", "cpu")
    if want == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FaceParser:
    """BiSeNet face parsing (19 CelebAMask-HQ classes). Shared by every backend."""

    def __init__(self, weights: Path = BISENET_WEIGHTS) -> None:
        import torch

        from .vendor.csd_mt import BiSeNet

        if not weights.exists():
            raise FileNotFoundError(missing_weights_message())
        self.device = _device()
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net.eval().to(self.device)

    def parse(self, rgb: np.ndarray) -> np.ndarray:
        """19-class label map (uint8) at the input's resolution."""
        import torch

        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (PARSE_SIZE, PARSE_SIZE), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32)
        t = torch.from_numpy(np.transpose(x, (2, 0, 1))[None]).to(self.device)
        with torch.no_grad():
            out = self.net(t)[0]
        labels = out[0].argmax(0).cpu().numpy().astype(np.uint8)
        return cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)


@lru_cache(maxsize=1)
def get_parser() -> FaceParser:
    return FaceParser()


class CSDMTEngine:
    """CSD-MT generator (~28 MB). Loaded once (``get_engine``). Fast, CPU-friendly, 256 px."""

    name = "csd-mt"
    size = NET_SIZE
    framing = "tight"

    def __init__(self, weights: Path = CSDMT_WEIGHTS) -> None:
        import torch

        from .vendor.csd_mt import Generator

        if not weights.exists():
            raise FileNotFoundError(missing_weights_message())
        self.device = _device()
        self.parser = get_parser()

        self.gen = Generator(input_dim=3, parse_dim=10, ngf=16, device=self.device)
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        self.gen.load_state_dict(ckpt["gen"])
        self.gen.eval().to(self.device)

    def parse(self, rgb: np.ndarray) -> np.ndarray:
        return self.parser.parse(rgb)

    @staticmethod
    def _split_parse(parse: np.ndarray) -> np.ndarray:
        """CSD-MT's 10-channel semantic encoding of the 19 BiSeNet labels (H, W, 10).

        One deliberate departure from upstream: spectacles are grouped with *background*, not
        skin. Upstream lets the attention sample makeup from under a reference's glasses, which
        transfers the dark frames as eye shadow; as background they are simply not a source.
        """
        groups = [
            [_BG, _CLOTH, _HAIR, _HAT, _EARRING, _GLASSES],
            [_SKIN],
            [_L_BROW, _R_BROW],
            [_L_EYE, _R_EYE],
            [_L_EAR, _R_EAR],
            [_NOSE],
            [_MOUTH],
            [_U_LIP],
            [_L_LIP],
            [_NECK, _NECKLACE],
        ]
        out = np.zeros(parse.shape + (10,), np.float32)
        for ch, labels in enumerate(groups):
            out[..., ch] = np.isin(parse, labels)
        return out

    @staticmethod
    def _all_mask(split: np.ndarray) -> np.ndarray:
        """Face region the generator edits: everything but background/hair, eyes, mouth."""
        m = (split[..., 0] == 0) & (split[..., 3] == 0) & (split[..., 6] == 0)
        return m.astype(np.float32)

    def _pack(self, rgb256: np.ndarray, parse256: np.ndarray):
        import torch

        split = self._split_parse(parse256)
        mask = self._all_mask(split)
        img = np.transpose(rgb256.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None].to(self.device)  # noqa: E731
        return to(img), to(np.transpose(split, (2, 0, 1))), to(np.repeat(mask[None], 3, 0))

    def transfer_crop(
        self, target_rgb: np.ndarray, reference_rgb: np.ndarray,
        target_parse: np.ndarray | None = None, **_,
    ) -> np.ndarray:
        """Render the reference's makeup onto the target crop. Both RGB uint8 ``NET_SIZE``²."""
        import torch

        if target_parse is None:
            target_parse = self.parse(target_rgb)
        t_img, t_parse, t_mask = self._pack(target_rgb, target_parse)
        r_img, r_parse, r_mask = self._pack(reference_rgb, self.parse(reference_rgb))
        with torch.no_grad():
            out = self.gen(source_img=t_img, source_parse=t_parse, source_all_mask=t_mask,
                           ref_img=r_img, ref_parse=r_parse, ref_all_mask=r_mask)
        y = out["transfer_img"][0, :3].cpu().float().numpy()
        y = np.transpose((y / 2.0 + 0.5) * 255.0, (1, 2, 0))
        return np.clip(y + 0.5, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def get_engine() -> CSDMTEngine:
    return CSDMTEngine()


BACKENDS = ("csd-mt", "diffusion")


def get_backend(backend: str = "csd-mt"):
    """The engine object for a backend name: ``"csd-mt"`` (fast GAN, CPU) or ``"diffusion"``
    (Stable-Makeup on SD 1.5 — generates the face anew; needs MPS/CUDA to be practical)."""
    if backend == "csd-mt":
        return get_engine()
    if backend == "diffusion":
        from . import diffusion

        return diffusion.get_engine()
    raise ValueError(f"Unknown backend '{backend}' (have: {', '.join(BACKENDS)})")


def backend_available(backend: str = "csd-mt") -> tuple[bool, str]:
    """(ok, reason-if-not) without loading any model."""
    if not BISENET_WEIGHTS.exists():
        return False, missing_weights_message()
    if backend == "csd-mt":
        return (True, "") if CSDMT_WEIGHTS.exists() else (False, missing_weights_message())
    if backend == "diffusion":
        from . import diffusion

        if not diffusion.available():
            return False, diffusion.missing_weights_message()
        try:
            import diffusers  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "The diffusion engine needs: pip install diffusers transformers accelerate matplotlib"
        return True, ""
    return False, f"Unknown backend '{backend}'"


# --- region weights + composite ----------------------------------------------------------------
def region_weights(parse: np.ndarray, strengths: dict[str, float]) -> np.ndarray:
    """Per-pixel strength map in [0, ~1] from the parse map and the four region sliders.

    Regions: lips (both lips), brows, eyes (a halo around the eye openings — shadow + liner),
    skin (the rest of the face + nose + neck). Each is feathered so neighbouring strengths blend.
    """
    h, w = parse.shape
    px = max(min(h, w) // 64, 1)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731

    lips = np.isin(parse, [_U_LIP, _L_LIP]).astype(np.uint8)
    brows = np.isin(parse, [_L_BROW, _R_BROW]).astype(np.uint8)
    eyes_open = np.isin(parse, [_L_EYE, _R_EYE]).astype(np.uint8)
    eyes = cv2.dilate(eyes_open, k(6 * px)) & ~brows & ~lips
    face = np.isin(parse, [_SKIN, _NOSE, _L_BROW, _R_BROW, _U_LIP, _L_LIP,
                           _L_EYE, _R_EYE]).astype(np.uint8)
    skin = face & ~lips & ~brows & ~eyes

    total = np.zeros((h, w), np.float32)
    norm = np.zeros((h, w), np.float32)
    for name, m in (("lips", lips), ("brows", brows), ("eyes", eyes), ("skin", skin)):
        soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5 * px)
        total += soft * float(strengths.get(name, 1.0))
        norm += soft
    return total / np.maximum(norm, 1e-6)


def edit_mask(parse: np.ndarray) -> np.ndarray:
    """Where the delta may land: face skin + makeup regions, feathered; not hair, ears, neck,
    the eyes' interior (iris/sclera), the mouth opening, or background. The neck is left alone on
    purpose: foundation stops at the jaw in real life, and a lightened neck above an untouched
    chest is the classic "mask" tell."""
    h, w = parse.shape
    px = max(min(h, w) // 64, 1)
    keep = np.isin(parse, [_SKIN, _NOSE, _L_BROW, _R_BROW, _U_LIP, _L_LIP]).astype(np.uint8)
    # Eye openings and the mouth interior are excluded, but a thin feathered edge lets liner
    # and lip-line pigment land right at the border instead of stopping short.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k)
    soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2.0 * px)
    return np.clip(soft, 0.0, 1.0)


@dataclass
class TransferResult:
    image: np.ndarray                 # full-resolution BGR result
    target: np.ndarray                # original BGR photo
    crop: Crop
    net_in: np.ndarray                # RGB target crop fed to the network (engine.size px)
    net_out: np.ndarray               # RGB network output, same size
    reference_crop: np.ndarray        # RGB reference crop, same size
    mask: np.ndarray                  # (crop.size, crop.size) float composite gate
    backend: str = "csd-mt"


def _reference_crop_rgb(
    reference: str | Path | np.ndarray, size: int, framing: str = "tight",
) -> np.ndarray:
    """The reference face, framed for an engine, as RGB uint8 ``size``².

    A reference is a photo path or an RGB array (a stored look). Either way the face is
    detected and cropped with the engine's own framing, so one stored look serves both the
    tight-crop GAN and the portrait-crop diffusion model. An array with no detectable face is
    centre-cropped and resized as-is.
    """
    if isinstance(reference, np.ndarray):
        bgr = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
        try:
            face = detect_face_bgr(bgr, name="reference")
        except ValueError:
            s_ = min(bgr.shape[:2])
            y0, x0 = (bgr.shape[0] - s_) // 2, (bgr.shape[1] - s_) // 2
            sq = bgr[y0:y0 + s_, x0:x0 + s_]
            interp = cv2.INTER_AREA if s_ > size else cv2.INTER_CUBIC
            return cv2.cvtColor(cv2.resize(sq, (size, size), interpolation=interp),
                                cv2.COLOR_BGR2RGB)
    else:
        face = detect_face(reference)
    bgr = crop_image(face.image, face_crop(face, framing), size)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def remove_glasses(rgb: np.ndarray) -> np.ndarray:
    """Paint spectacle frames out of a reference crop (inpainting from the surrounding skin).

    A generative engine reads the *whole* reference as "the look", so frames come back drawn on
    the target as dark rings. BiSeNet's glasses class marks the whole spectacle area, eyes and
    brows included, so only its *dark* pixels outside the landmark eye/brow regions are treated
    as frame — eye makeup behind the lenses survives. Best effort: a no-op when the reference
    has no glasses or no detectable landmarks."""
    parse = get_parser().parse(rgb)
    glasses = (parse == _GLASSES).astype(np.uint8)
    if glasses.sum() < 0.002 * glasses.size:
        return rgb
    try:
        face = detect_face_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), name="reference")
    except ValueError:
        return rgb
    px = max(rgb.shape[0] // 128, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    protect = np.zeros_like(glasses)
    for name in ("left_eye", "right_eye", "left_brow", "right_brow"):
        protect |= (face.masks[name] > 0).astype(np.uint8)
    protect = cv2.dilate(protect, k, iterations=2)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    skin = np.isin(parse, [_SKIN, _NOSE])
    v_skin = float(np.median(gray[skin])) if skin.sum() > 50 else 160.0
    frames = glasses & (gray < 0.55 * v_skin).astype(np.uint8) & (1 - protect)
    if frames.sum() < 0.0005 * frames.size:
        return rgb
    frames = cv2.dilate(frames, k, iterations=2) & (1 - protect)
    return cv2.inpaint(rgb, frames * 255, 3 * px, cv2.INPAINT_TELEA)


def reference_from_photo(photo: str | Path, size: int = REFERENCE_SIZE) -> np.ndarray:
    """Detect the face in a makeup photo and return its square RGB crop (what a look stores).
    Stored with the looser *portrait* framing; each engine re-crops from it as it needs."""
    return _reference_crop_rgb(photo, size, "portrait")


def _damp_complexion(
    base_bgr: np.ndarray, delta: np.ndarray, parse: np.ndarray, keep: float,
) -> np.ndarray:
    """Limit how much of the reference's *complexion* (foundation, exposure, skin lightening)
    lands on the target's skin, while keeping skin *color* makeup (blush, bronzer) in full.

    Works in Lab on bare skin (not lips/brows/eye band). The network's change there is split
    into what foundation does and what makeup does:

    - **L** (lightness): scaled to ``keep`` — whitening / exposure lift is foundation.
    - **a / b** uniform cast (mean over skin): scaled to ``keep`` — one colour cast over the
      whole face is foundation tint.
    - **a / b** local variation: kept, except a *cool* local shift (−b, toward blue/grey),
      which is scaled to ``keep`` — blush and bronzer are warm; a grey-blue patch on skin is
      a pale reference's foundation or a GAN reconstruction error, never makeup.

    ``keep=1`` is the raw network output, ``keep=0`` leaves the target's own skin tone and
    shading untouched. Feathered so the regions hand off smoothly.
    """
    if keep >= 1.0:
        return delta
    from skimage import color

    h, w = parse.shape
    px = max(min(h, w) // 64, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12 * px + 1, 12 * px + 1))
    eyes = cv2.dilate(np.isin(parse, [_L_EYE, _R_EYE]).astype(np.uint8), k)
    skin = (np.isin(parse, [_SKIN, _NOSE]).astype(np.uint8) & ~eyes).astype(bool)
    if skin.sum() < 16:
        return delta

    base = base_bgr.astype(np.float32)
    lab0 = color.rgb2lab(cv2.cvtColor(base, cv2.COLOR_BGR2RGB) / 255.0)
    lab1 = color.rgb2lab(cv2.cvtColor(np.clip(base + delta, 0, 255), cv2.COLOR_BGR2RGB) / 255.0)
    d = (lab1 - lab0).astype(np.float32)

    d_skin = d.copy()
    d_skin[..., 0] *= keep
    cast = d[skin][:, 1:].mean(axis=0)
    local = d[..., 1:] - cast[None, None, :]
    local_b = local[..., 1]
    local[..., 1] = np.where(local_b < 0, local_b * keep, local_b)
    d_skin[..., 1:] = keep * cast[None, None, :] + local

    soft = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 2.0 * px)[..., None]
    d_new = d + soft * (d_skin - d)
    rgb = np.clip(color.lab2rgb((lab0 + d_new).astype(np.float64)), 0.0, 1.0)
    new = cv2.cvtColor((rgb * 255.0).astype(np.float32), cv2.COLOR_RGB2BGR)
    return new - base


def _border_fade(size: int, frac: float = 0.06) -> np.ndarray:
    """(size, size) ramp from 0 at the crop border to 1 inside. A tight crop can cut through
    the forehead or jaw; without this the edit stops dead at the crop edge and shows as a band."""
    n = max(int(round(frac * size)), 1)
    ramp = np.minimum(np.arange(size, dtype=np.float32) + 0.5, n) / n
    edge = np.minimum(ramp, ramp[::-1])
    return np.minimum.outer(edge, edge)


DEFAULT_COMPLEXION = 0.4   # keep 40% of the reference's overall skin-tone shift by default

# Network outputs keyed by (backend, inputs, sampling options). The composite is cheap and the
# network is not (a minute for diffusion), so moving a strength slider must not re-run it.
_NET_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_NET_CACHE_MAX = 8


def _cache_key(backend: str, net_in: np.ndarray, ref: np.ndarray, opts: dict) -> tuple:
    h = hashlib.sha1(net_in.tobytes())
    h.update(ref.tobytes())
    return (backend, h.hexdigest(), tuple(sorted(opts.items())))


def transfer_makeup(
    target: str | Path | FaceData,
    reference: str | Path | np.ndarray,
    *,
    strengths: dict[str, float] | None = None,
    complexion: float = DEFAULT_COMPLEXION,
    occlusion: bool = True,
    backend: str = "csd-mt",
    steps: int | None = None,
    guidance: float | None = None,
    strength: float | None = None,
    seed: int = 0,
    progress=None,
    engine=None,
) -> TransferResult:
    """Put the reference's makeup on the target photo at full resolution.

    ``backend`` picks the network: ``"csd-mt"`` (fast GAN) or ``"diffusion"`` (Stable-Makeup;
    ``steps`` / ``guidance`` / ``strength`` / ``seed`` / ``progress`` apply to it only). ``strengths`` scales
    the makeup per region (``DEFAULT_STRENGTHS`` keys); values above 1 exaggerate, 0 removes.
    ``complexion`` (0..1) is how much of the reference's *overall* skin-tone shift to keep (see
    ``_damp_complexion``). ``occlusion`` gates the result by the segmenter's skin mask so hair
    and hands crossing the face stay clean.
    """
    engine = engine or get_backend(backend)
    parser = get_parser()
    face = target if isinstance(target, FaceData) else detect_face(target)
    strengths = {**DEFAULT_STRENGTHS, **(strengths or {})}
    size = engine.size

    crop = face_crop(face, engine.framing)
    ys, xs = crop.slice()
    crop_bgr = face.image[ys, xs]
    net_in = cv2.cvtColor(crop_image(face.image, crop, size), cv2.COLOR_BGR2RGB)
    ref_rgb = _reference_crop_rgb(reference, size, engine.framing)
    if engine.name != "csd-mt":
        ref_rgb = remove_glasses(ref_rgb)

    # Parse once at crop resolution; the GAN gets its own network-size copy.
    parse_full = parser.parse(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    opts = {}
    if engine.name != "csd-mt":
        opts = {"seed": int(seed)}
        if steps is not None:
            opts["steps"] = int(steps)
        if guidance is not None:
            opts["guidance"] = float(guidance)
        if strength is not None:
            opts["strength"] = float(strength)
    key = _cache_key(engine.name, net_in, ref_rgb, opts)
    if (net_out := _NET_CACHE.get(key)) is not None:
        _NET_CACHE.move_to_end(key)
    else:
        parse_net = cv2.resize(parse_full, (size, size), interpolation=cv2.INTER_NEAREST)
        landmarks = (face.landmarks - np.array([crop.x0, crop.y0], np.float32)) * (size / crop.size)
        net_out = engine.transfer_crop(net_in, ref_rgb, target_parse=parse_net,
                                       landmarks=landmarks, progress=progress, **opts)
        _NET_CACHE[key] = net_out
        while len(_NET_CACHE) > _NET_CACHE_MAX:
            _NET_CACHE.popitem(last=False)

    # Delta composite: only the makeup *change* is upsampled and added to the untouched photo.
    delta = net_out.astype(np.float32) - net_in.astype(np.float32)
    delta = cv2.resize(delta, (crop.size, crop.size), interpolation=cv2.INTER_CUBIC)
    scale = crop.size / size
    if scale > 1.5:
        # Upsampled network pixels would otherwise show as blocky edges on lip/liner borders.
        delta = cv2.GaussianBlur(delta, (0, 0), 0.5 * scale)
    delta = cv2.cvtColor(delta, cv2.COLOR_RGB2BGR)
    delta = _damp_complexion(crop_bgr, delta, parse_full, complexion)

    gate = edit_mask(parse_full) * region_weights(parse_full, strengths) * _border_fade(crop.size)
    if occlusion and (skin := face.skin_visibility) is not None:
        gate = gate * skin[ys, xs]

    out = face.image.copy()
    region = crop_bgr.astype(np.float32) + gate[..., None] * delta
    out[ys, xs] = np.clip(region, 0, 255).astype(np.uint8)
    return TransferResult(image=out, target=face.image, crop=crop, net_in=net_in,
                          net_out=net_out, reference_crop=ref_rgb, mask=gate,
                          backend=engine.name)


# --- looks ------------------------------------------------------------------------------------
def create_look(photo: str | Path, name: str, *, store=None, notes: str = ""):
    """Turn one photo of someone wearing makeup into a saved look. Returns ``(reference_rgb,
    look_dir)``. Raises ``ValueError`` when no face is found."""
    from .storage import LookStore

    store = store or LookStore()
    ref = reference_from_photo(photo)
    meta = {
        "source_with": str(photo),
        "blend": "engine",
        "blend_params": {
            "regions": {k: {"strength": v} for k, v in DEFAULT_STRENGTHS.items()},
            "complexion": DEFAULT_COMPLEXION,
        },
        "notes": notes,
    }
    return ref, store.save_reference(name, ref, meta)


def reference_swatches(reference_rgb: np.ndarray) -> dict[str, str]:
    """Mean colour (``#rrggbb``) of the reference's lips, eye band, brows and skin — the look's
    palette, for display. Empty dict if the parser finds no face."""
    rgb = _reference_crop_rgb(reference_rgb, NET_SIZE)
    parse = get_parser().parse(rgb)
    px = max(NET_SIZE // 64, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12 * px + 1, 12 * px + 1))
    eyes_open = np.isin(parse, [_L_EYE, _R_EYE]).astype(np.uint8)
    brows = np.isin(parse, [_L_BROW, _R_BROW])
    regions = {
        "lips": np.isin(parse, [_U_LIP, _L_LIP]),
        "eyes": cv2.dilate(eyes_open, k).astype(bool) & ~eyes_open.astype(bool) & ~brows,
        "brows": brows,
        "skin": np.isin(parse, [_SKIN, _NOSE]),
    }
    out: dict[str, str] = {}
    for name, m in regions.items():
        if m.sum() < 20:
            continue
        r, g, b = rgb[m].reshape(-1, 3).mean(axis=0)
        out[name] = "#{:02x}{:02x}{:02x}".format(*(int(np.clip(c, 0, 255)) for c in (r, g, b)))
    return out


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Reference-based makeup transfer (CSD-MT engine).")
    ap.add_argument("target", help="photo to put makeup on")
    ap.add_argument("reference", help="photo of someone wearing the makeup")
    ap.add_argument("--out", default="preprocess_out/engine")
    ap.add_argument("--strength", action="append", metavar="REGION=X",
                    help="per-region strength, e.g. --strength lips=1.2 (repeatable)")
    ap.add_argument("--complexion", type=float, default=DEFAULT_COMPLEXION,
                    help="0..1: how much of the reference's overall skin-tone shift to keep")
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--backend", default="csd-mt", choices=BACKENDS,
                    help="csd-mt = fast GAN (CPU); diffusion = Stable-Makeup on SD 1.5")
    ap.add_argument("--steps", type=int, default=None, help="diffusion: denoising steps")
    ap.add_argument("--guidance", type=float, default=None,
                    help="diffusion: makeup guidance (1.1 light .. 2.0 heavy; default 1.6)")
    ap.add_argument("--denoise", type=float, default=None,
                    help="diffusion: 1.0 generates from noise; lower anchors to your photo")
    ap.add_argument("--seed", type=int, default=0, help="diffusion: noise seed")
    args = ap.parse_args()

    strengths = {}
    for item in args.strength or []:
        name, _, val = item.partition("=")
        strengths[name.strip()] = float(val)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = transfer_makeup(args.target, args.reference, strengths=strengths,
                          complexion=args.complexion, occlusion=not args.no_occlusion,
                          backend=args.backend, steps=args.steps, guidance=args.guidance,
                          strength=args.denoise, seed=args.seed,
                          progress=lambda i, n: print(f"\r  step {i}/{n}", end="", flush=True))
    stem = f"{Path(args.target).stem}_x_{Path(args.reference).stem}"
    if args.backend != "csd-mt":
        print()
        stem += f"_{args.backend}"
    cv2.imwrite(str(out / f"{stem}.png"), res.image)
    s = 384 if args.backend != "csd-mt" else NET_SIZE
    rs = lambda im: cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA)  # noqa: E731
    strip = np.hstack([
        rs(cv2.cvtColor(res.net_in, cv2.COLOR_RGB2BGR)),
        rs(cv2.cvtColor(res.reference_crop, cv2.COLOR_RGB2BGR)),
        rs(cv2.cvtColor(res.net_out, cv2.COLOR_RGB2BGR)),
        rs(res.image[res.crop.slice()]),
        cv2.cvtColor((cv2.resize(res.mask, (s, s)) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
    ])
    cv2.imwrite(str(out / f"{stem}_strip.png"), strip)
    print(f"Wrote {out / f'{stem}.png'} (+ _strip.png: target | reference | net | composite | gate)")


if __name__ == "__main__":
    _main()
