"""Fetch the makeup-engine weights into ``weights/`` (``fetch_weights.py``).

Two files, ~80 MB total, neither tracked in git:

- ``csd_mt.pth`` — CSD-MT generator (CVPR 2024), from the authors' Google Drive folder.
- ``bisenet_79999_iter.pth`` — BiSeNet face parser (face-parsing.PyTorch), same folder; also
  mirrored on the Hugging Face Hub (``qihoo360/FLUX-Makeup``) which is tried first because it
  needs no Drive quota.

Run::

    python -m makeup_transfer.fetch_weights
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
