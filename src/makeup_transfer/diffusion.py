"""Diffusion makeup engine: Stable-Makeup (SIGGRAPH 2025) on Stable Diffusion 1.5 (``diffusion.py``).

Instead of editing the photo, this engine **generates a new 512 px face** conditioned on three
things, then hands it to ``engine.transfer_makeup`` for the usual full-resolution composite:

- the **target face** (identity / content) through a ControlNet-style *content encoder*,
- a **face-structure drawing** (68 landmarks as coloured contours) through a second ControlNet,
- the **reference makeup** as multi-layer CLIP ViT-L features fed to the UNet's cross-attention
  (the *detail-preserving makeup encoder*) in place of a text prompt.

This is a port of the authors' inference path (https://github.com/Xiaojiu-z/Stable-Makeup,
Apache-2.0) onto current ``diffusers``/``transformers`` — their repo pins a bundled 2023
``diffusers`` fork. Two substitutions: the SPIGA landmark detector is replaced by the MediaPipe
landmarks the pipeline already has (mapped to the 68-point layout the structure encoder was
trained on, and drawn with the same matplotlib recipe), and their 1.8k-line pipeline is the
~40-line denoising loop in ``StableMakeupEngine.transfer_crop``.

Heavy: ~2 B parameters (UNet + two ControlNets + CLIP-L). Runs on Apple-silicon MPS or CUDA in
fp16 (about a minute on an M2 at 30 steps); CPU works but takes several minutes.

Weights (not tracked): ``python -m makeup_transfer.fetch_weights --diffusion``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "stablemakeup"
MAKEUP_ENCODER = _WEIGHTS / "pytorch_model.bin"
ID_ENCODER = _WEIGHTS / "pytorch_model_1.bin"
POSE_ENCODER = _WEIGHTS / "pytorch_model_2.bin"

SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CLIP_REPO = "openai/clip-vit-large-patch14"

SIZE = 512
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 1.6   # authors' default; ~1.1 for light makeup, ~2 for heavy
# 1.0 = generate from pure noise (upstream behaviour: strongest makeup, most identity drift).
# Below 1 the sampler starts from the *noised target face* instead, so face shape, shading and
# skin tone are anchored to the real photo and only the remaining steps are run. 0.7 was
# picked by eye on a phone portrait: identity holds, lipstick/shadow/blush still land fully.
DEFAULT_STRENGTH = 0.7

# MediaPipe face-mesh indices for the 68 iBUG/300W landmarks, in 300W order:
# jaw (17), brows (5+5), nose bridge (4), nose base (5), eyes (6+6), outer lips (12), inner (8).
MP_TO_68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
]


def available() -> bool:
    """Are the Stable-Makeup weights present (cheap check, no torch import)?"""
    return MAKEUP_ENCODER.exists() and ID_ENCODER.exists() and POSE_ENCODER.exists()


def missing_weights_message() -> str:
    return ("Diffusion engine weights not found. Fetch them (about 7 GB) with:\n"
            "  python -m makeup_transfer.fetch_weights --diffusion")


def pick_device() -> str:
    """``MAKEUP_DIFFUSION_DEVICE`` if set, else CUDA > MPS > CPU."""
    import torch

    want = os.environ.get("MAKEUP_DIFFUSION_DEVICE", "")
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- structure drawing -------------------------------------------------------------------------
def structure_image(landmarks68: np.ndarray, size: int = SIZE) -> np.ndarray:
    """The structure encoder's conditioning image: 68 landmarks drawn as coloured contours on
    black (RGB uint8). Mirrors Stable-Makeup's ``conditioning_from_landmarks`` — same matplotlib
    primitives, colours and 4 pt line width — because the encoder was trained on exactly that
    rendering."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    pts = [(float(x), float(y)) for x, y in landmarks68]

    def patch(seg, color, closed=False):
        contour = list(seg)
        ops = [MplPath.MOVETO] + [MplPath.LINETO] * (len(contour) - 1)
        face = (0, 0, 0, 0)
        if closed:
            contour.append(contour[0])
            ops.append(MplPath.CLOSEPOLY)
            face = color
        return PathPatch(MplPath(contour, ops), facecolor=face, edgecolor=color, lw=4)

    dpi = 72
    fig = Figure(figsize=[size / dpi, size / dpi], dpi=dpi, tight_layout={"pad": 0})
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.imshow(np.zeros((size, size, 3)))
    for seg, color, closed in (
        (pts[0:17], "lime", False), (pts[17:22], "yellow", False), (pts[22:27], "yellow", False),
        (pts[27:31], "orange", False), (pts[31:36], "orange", False),
        (pts[36:42], "magenta", True), (pts[42:48], "magenta", True),
        (pts[48:60], "cyan", True), (pts[60:68], "blue", True),
    ):
        ax.add_patch(patch(seg, color, closed))
    ax.axis("off")
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3]
    if buf.shape[0] != size:  # tight_layout rounding on some backends
        import cv2

        buf = cv2.resize(buf, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(buf)


# --- the model ---------------------------------------------------------------------------------
def _ssr_processor_cls():
    import torch
    import torch.nn.functional as F
    from torch import nn

    class SSRAttnProcessor(nn.Module):
        """Cross-attention that reads the makeup tokens through its own key/value projections
        (``to_k_SSR`` / ``to_v_SSR``), reusing the UNet's query and output projections."""

        def __init__(self, hidden_size: int, cross_attention_dim: int = 1024):
            super().__init__()
            self.to_k_SSR = nn.Linear(cross_attention_dim, hidden_size, bias=False)
            self.to_v_SSR = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                    temb=None, **_):
            residual = hidden_states
            b = hidden_states.shape[0]
            query = attn.to_q(hidden_states)
            key = self.to_k_SSR(encoder_hidden_states)
            value = self.to_v_SSR(encoder_hidden_states)
            head_dim = key.shape[-1] // attn.heads
            split = lambda x: x.view(b, -1, attn.heads, head_dim).transpose(1, 2)  # noqa: E731
            out = F.scaled_dot_product_attention(split(query), split(key), split(value))
            out = out.transpose(1, 2).reshape(b, -1, attn.heads * head_dim).to(query.dtype)
            out = attn.to_out[1](attn.to_out[0](out))
            if attn.residual_connection:
                out = out + residual
            return out / attn.rescale_output_factor

    return SSRAttnProcessor


def _resampler():
    import torch.nn.functional as F
    from torch import nn

    class GEGLU(nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.proj = nn.Linear(dim_in, dim_out * 2)

        def forward(self, x):
            x, gate = self.proj(x).chunk(2, dim=-1)
            return x * F.gelu(gate)

    class FeedForward(nn.Module):
        def __init__(self, dim, mult=4):
            super().__init__()
            self.net = nn.Sequential(GEGLU(dim, dim * mult), nn.Dropout(0.0),
                                     nn.Linear(dim * mult, dim))

        def forward(self, x):
            return self.net(x)

    class SelfAttention(nn.Module):
        def __init__(self, dim, heads=8, dim_head=64):
            super().__init__()
            inner = heads * dim_head
            self.heads = heads
            self.to_q = nn.Linear(dim, inner, bias=False)
            self.to_k = nn.Linear(dim, inner, bias=False)
            self.to_v = nn.Linear(dim, inner, bias=False)
            self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(0.0))

        def forward(self, x):
            b, n, _ = x.shape
            split = lambda t: t.view(b, n, self.heads, -1).transpose(1, 2)  # noqa: E731
            out = F.scaled_dot_product_attention(
                split(self.to_q(x)), split(self.to_k(x)), split(self.to_v(x)))
            return self.to_out(out.transpose(1, 2).reshape(b, n, -1))

    class Resampler(nn.Module):
        def __init__(self, dim=1024):
            super().__init__()
            self.attn = SelfAttention(dim)
            self.ff = FeedForward(dim)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            return x + self.ff(self.norm2(x))

    return Resampler()


class StableMakeupEngine:
    """Stable-Makeup on SD 1.5. Loaded once (``get_engine``)."""

    name = "stable-makeup"
    size = SIZE
    framing = "portrait"

    def __init__(self, device: str | None = None) -> None:
        import torch
        from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel

        if not available():
            raise FileNotFoundError(missing_weights_message())
        self.device = device or pick_device()
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        dev, dt = self.device, self.dtype

        unet = UNet2DConditionModel.from_pretrained(
            SD15_REPO, subfolder="unet", variant="fp16", torch_dtype=dt)
        self.id_encoder = ControlNetModel.from_unet(unet)
        self.pose_encoder = ControlNetModel.from_unet(unet)
        self.id_encoder.load_state_dict(torch.load(ID_ENCODER, map_location="cpu", weights_only=True), strict=False)
        self.pose_encoder.load_state_dict(torch.load(POSE_ENCODER, map_location="cpu", weights_only=True), strict=False)

        makeup = torch.load(MAKEUP_ENCODER, map_location="cpu", weights_only=True)
        self.image_encoder = CLIPVisionModel.from_pretrained(CLIP_REPO)
        enc = {k[len("image_encoder."):]: v for k, v in makeup.items()
               if k.startswith("image_encoder.")}
        if enc:
            self.image_encoder.load_state_dict(enc, strict=False)
        self.clip_processor = CLIPImageProcessor()

        self.resampler = _resampler()
        self.resampler.load_state_dict(
            {k[len("resampler."):]: v for k, v in makeup.items() if k.startswith("resampler.")})

        # Makeup cross-attention: the checkpoint indexes its layers by position in
        # ``unet.attn_processors`` (self-attention entries included, parameter-free).
        SSR = _ssr_processor_cls()
        procs, loaded = {}, 0
        for i, (name, default) in enumerate(unet.attn_processors.items()):
            if name.endswith("attn1.processor"):
                procs[name] = default
                continue
            if name.startswith("mid_block"):
                hidden = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                hidden = list(reversed(unet.config.block_out_channels))[int(name[len("up_blocks.")])]
            else:
                hidden = unet.config.block_out_channels[int(name[len("down_blocks.")])]
            proc = SSR(hidden)
            proc.load_state_dict({
                "to_k_SSR.weight": makeup[f"SSR_layers.{i}.to_k_SSR.weight"],
                "to_v_SSR.weight": makeup[f"SSR_layers.{i}.to_v_SSR.weight"],
            })
            procs[name] = proc.to(dev, dt)
            loaded += 1
        if loaded != 16:
            raise RuntimeError(f"expected 16 makeup cross-attention layers, mapped {loaded}")
        unet.set_attn_processor(procs)
        del makeup

        self.unet = unet.to(dev).eval()
        self.id_encoder.to(dev, dt).eval()
        self.pose_encoder.to(dev, dt).eval()
        self.image_encoder.to(dev, dt).eval()
        self.resampler.to(dev, dt).eval()
        # The VAE stays float32: fp16 decoding on MPS can band or NaN.
        self.vae = AutoencoderKL.from_pretrained(
            SD15_REPO, subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(dev).eval()
        self.scheduler = DDIMScheduler.from_pretrained(SD15_REPO, subfolder="scheduler")

        # The two encoders were trained with the empty-prompt text embedding; compute it once.
        tok = CLIPTokenizer.from_pretrained(SD15_REPO, subfolder="tokenizer")
        text = CLIPTextModel.from_pretrained(SD15_REPO, subfolder="text_encoder", variant="fp16")
        ids = tok("", max_length=tok.model_max_length, padding="max_length", truncation=True,
                  return_tensors="pt").input_ids
        with torch.no_grad():
            self.null_text = text(ids)[0].to(dev, dt)
        del text, tok

    # -----------------------------------------------------------------------------------------
    def _makeup_tokens(self, reference_rgb: np.ndarray):
        """(uncond, cond) makeup token sequences: 12 CLIP layers x 257 tokens, resampled."""
        import torch
        from PIL import Image

        px = self.clip_processor(images=Image.fromarray(reference_rgb), return_tensors="pt")
        px = px.pixel_values.to(self.device, self.dtype)

        def tokens(x):
            hs = self.image_encoder(x, output_hidden_states=True).hidden_states[2::2]
            return self.resampler(torch.cat(hs, dim=1))

        return tokens(torch.zeros_like(px)), tokens(px)

    def _cond(self, rgb: np.ndarray):
        import torch

        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255.0
        return t.to(self.device, self.dtype).repeat(2, 1, 1, 1)

    def transfer_crop(
        self,
        target_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        *,
        landmarks: np.ndarray | None = None,
        steps: int = DEFAULT_STEPS,
        guidance: float = DEFAULT_GUIDANCE,
        strength: float = DEFAULT_STRENGTH,
        seed: int = 0,
        progress=None,
        **_,
    ) -> np.ndarray:
        """Generate the target face wearing the reference's makeup. Both inputs RGB uint8
        ``SIZE``²; ``landmarks`` are the target's 468 MediaPipe points in crop pixels (without
        them the structure encoder gets a black image, as upstream does when no face is found).
        ``strength`` < 1 anchors the generation to the target photo (see ``DEFAULT_STRENGTH``).
        ``progress(step, total)`` is called after each denoising step."""
        import torch

        if landmarks is not None:
            pose = structure_image(np.asarray(landmarks)[MP_TO_68], self.size)
        else:
            pose = np.zeros((self.size, self.size, 3), np.uint8)
        self.last_structure = pose

        with torch.inference_mode():
            uncond, cond = self._makeup_tokens(reference_rgb)
            tokens = torch.cat([uncond, cond])
            null = self.null_text.repeat(2, 1, 1)
            conds = ((self.id_encoder, self._cond(target_rgb)), (self.pose_encoder, self._cond(pose)))

            self.scheduler.set_timesteps(steps)
            gen = torch.Generator("cpu").manual_seed(int(seed))
            latents = torch.randn((1, 4, self.size // 8, self.size // 8), generator=gen)
            timesteps = self.scheduler.timesteps
            strength = float(min(max(strength, 0.05), 1.0))
            if strength < 1.0:
                skip = min(int(round(steps * (1.0 - strength))), steps - 1)
                timesteps = timesteps[skip:]
                x0 = torch.from_numpy(np.ascontiguousarray(target_rgb)).permute(2, 0, 1)[None]
                x0 = (x0.float() / 127.5 - 1.0).to(self.device)
                init = self.vae.encode(x0).latent_dist.mean * self.vae.config.scaling_factor
                latents = self.scheduler.add_noise(init.cpu(), latents, timesteps[:1])
            else:
                latents = latents * self.scheduler.init_noise_sigma
            latents = latents.to(self.device, self.dtype)
            total = len(timesteps)

            for i, t in enumerate(timesteps):
                x = self.scheduler.scale_model_input(torch.cat([latents] * 2), t)
                down, mid = None, None
                for net, c in conds:
                    d, m = net(x, t, encoder_hidden_states=null, controlnet_cond=c,
                               conditioning_scale=1.0, return_dict=False)
                    down = d if down is None else [a + b for a, b in zip(down, d)]
                    mid = m if mid is None else mid + m
                eps = self.unet(x, t, encoder_hidden_states=tokens,
                                down_block_additional_residuals=down,
                                mid_block_additional_residual=mid, return_dict=False)[0]
                e_uncond, e_cond = eps.chunk(2)
                eps = e_uncond + guidance * (e_cond - e_uncond)
                latents = self.scheduler.step(eps, t, latents, return_dict=False)[0]
                if progress is not None:
                    progress(i + 1, total)

            img = self.vae.decode(latents.float() / self.vae.config.scaling_factor).sample
        img = (img[0].clamp(-1, 1) + 1) / 2
        return (img.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


@lru_cache(maxsize=1)
def get_engine() -> StableMakeupEngine:
    return StableMakeupEngine()
