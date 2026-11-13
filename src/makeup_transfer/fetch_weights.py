"""Fetch the makeup-engine weights into ``weights/`` (``fetch_weights.py``).

Two files, ~80 MB total, neither tracked in git:

- ``csd_mt.pth`` — CSD-MT generator (CVPR 2024), from the authors' Google Drive folder.
- ``bisenet_79999_iter.pth`` — BiSeNet face parser (face-parsing.PyTorch), same folder; also
  mirrored on the Hugging Face Hub (``qihoo360/FLUX-Makeup``) which is tried first because it
  needs no Drive quota.

``--diffusion`` additionally fetches the Stable-Makeup engine (``diffusion.py``), about 7 GB:

- ``weights/stablemakeup/pytorch_model{,_1,_2}.bin`` — makeup / content / structure encoders,
  from the Stable-Makeup authors' Google Drive folder (4.3 GB).
- Stable Diffusion 1.5 (fp16 UNet, VAE, text encoder) and CLIP ViT-L/14 into the Hugging Face
  cache (about 3.7 GB).

Run::

    python -m makeup_transfer.fetch_weights              # fast engine only
    python -m makeup_transfer.fetch_weights --diffusion  # + the diffusion engine
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from .engine import BISENET_WEIGHTS, CSDMT_WEIGHTS

CSDMT_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1pvSgkpsb7k6Ph1_oCmFkMQZPgL7PaTO0"
BISENET_HF = ("qihoo360/FLUX-Makeup", "79999_iter.pth")


def _fetch_bisenet_hf(dest: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            got = hf_hub_download(BISENET_HF[0], BISENET_HF[1], local_dir=tmp)
            shutil.copy2(got, dest)
        return True
    except Exception as e:  # noqa: BLE001 — fall through to Drive
        print(f"  HF mirror failed ({e}); trying Google Drive")
        return False


def _fetch_drive_folder(need: dict[str, Path]) -> None:
    """Download the CSD-MT release folder and copy the files we need into place."""
    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown is required: pip install gdown")
    with tempfile.TemporaryDirectory() as tmp:
        gdown.download_folder(CSDMT_DRIVE_FOLDER, output=tmp, quiet=False, use_cookies=False)
        for src_name, dest in need.items():
            hits = list(Path(tmp).rglob(src_name))
            if not hits:
                raise SystemExit(f"{src_name} not found in the downloaded folder")
            shutil.copy2(hits[0], dest)


STABLE_MAKEUP_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1397t27GrUyLPnj17qVpKWGwg93EcaFfg"
SD15_FILES = [
    "model_index.json", "scheduler/*", "tokenizer/*",
    "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
]
CLIP_FILES = ["config.json", "model.safetensors", "preprocessor_config.json"]


def fetch_diffusion(force: bool = False) -> None:
    """Stable-Makeup encoders (Drive) + SD 1.5 and CLIP ViT-L (Hugging Face cache)."""
    from . import diffusion

    names = {p.name: p for p in (diffusion.MAKEUP_ENCODER, diffusion.ID_ENCODER,
                                 diffusion.POSE_ENCODER)}
    need = {n: p for n, p in names.items() if force or not p.exists()}
    if need:
        try:
            import gdown
        except ImportError:
            raise SystemExit("gdown is required: pip install gdown")
        diffusion.MAKEUP_ENCODER.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            print(f"Stable-Makeup encoders -> {diffusion.MAKEUP_ENCODER.parent}")
            gdown.download_folder(STABLE_MAKEUP_DRIVE_FOLDER, output=tmp, quiet=False,
                                  use_cookies=False)
            for name, dest in need.items():
                hits = [h for h in Path(tmp).rglob(name)]
                if not hits:
                    raise SystemExit(f"{name} not found in the downloaded folder")
                shutil.move(str(hits[0]), dest)
    from huggingface_hub import snapshot_download

    print("Stable Diffusion 1.5 + CLIP ViT-L -> Hugging Face cache")
    snapshot_download(diffusion.SD15_REPO, allow_patterns=SD15_FILES)
    snapshot_download(diffusion.CLIP_REPO, allow_patterns=CLIP_FILES)
    for p in names.values():
        print(f"  {p.name}: {p.stat().st_size / 1e9:.2f} GB" if p.exists() else f"  {p.name}: MISSING")


def main(force: bool = False) -> None:
    CSDMT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    need: dict[str, Path] = {}
    if force or not BISENET_WEIGHTS.exists():
        print(f"BiSeNet -> {BISENET_WEIGHTS}")
        if not _fetch_bisenet_hf(BISENET_WEIGHTS):
            need["79999_iter.pth"] = BISENET_WEIGHTS
    if force or not CSDMT_WEIGHTS.exists():
        print(f"CSD-MT -> {CSDMT_WEIGHTS}")
        need["CSD_MT.pth"] = CSDMT_WEIGHTS
    if need:
        _fetch_drive_folder(need)
    for p in (CSDMT_WEIGHTS, BISENET_WEIGHTS):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB" if p.exists() else f"  {p.name}: MISSING")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
    if "--diffusion" in sys.argv:
        fetch_diffusion(force="--force" in sys.argv)
