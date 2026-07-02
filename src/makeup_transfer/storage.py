"""Fake, local-file storage for makeup "looks".

A *look* is the savable/shareable artifact: a UV-space makeup layer (RGBA PNG) plus metadata.
For now everything lives on disk under ``looks/<name>/``. This module is intentionally a thin
interface so a real backend (object store + DB) can replace it later WITHOUT touching the NN code.

Layout per look::

    looks/<name>/
        layer.png    # RGBA makeup layer in UV space
        meta.json    # {name, created, source_with, source_without, blend, notes}

Usage::

    store = LookStore()                      # defaults to <repo>/looks
    store.save("smoky-eye", layer_rgba, meta={"blend": "multiply"})
    layer = store.load_layer("smoky-eye")
    store.list_looks()                       # ["smoky-eye", ...]
    store.export_zip("smoky-eye", "out.zip") # "share" == zip the folder, for now
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# Repo root = three levels up from this file (src/makeup_transfer/storage.py).
_DEFAULT_LOOKS_DIR = Path(__file__).resolve().parents[2] / "looks"

_LAYER_FILE = "layer.png"
_META_FILE = "meta.json"


@dataclass
class LookMeta:
    """Metadata stored alongside a makeup layer."""

    name: str
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_with: str | None = None      # filename/path of the with-makeup source
    source_without: str | None = None   # filename/path of the without-makeup source
    blend: str = "multiply"             # how the layer should be composited onto a target
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LookMeta":
        known = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**known)


class LookStore:
    """Local-file implementation of the look store (the fake "backend")."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else _DEFAULT_LOOKS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    # --- internal helpers -------------------------------------------------
    def _look_dir(self, name: str) -> Path:
        safe = name.strip().replace("/", "_").replace("\\", "_")
        if not safe:
            raise ValueError("Look name must not be empty.")
        return self.root / safe

    @staticmethod
    def _to_rgba_image(layer: "np.ndarray | Image.Image") -> Image.Image:
        if isinstance(layer, Image.Image):
            return layer.convert("RGBA")
        arr = np.asarray(layer)
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError(f"Makeup layer must be RGBA (H, W, 4); got shape {arr.shape}.")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGBA")

    # --- public API -------------------------------------------------------
    def save(
        self,
        name: str,
        layer: "np.ndarray | Image.Image",
        meta: dict | LookMeta | None = None,
    ) -> Path:
        """Persist a makeup layer (RGBA) + metadata. Returns the look directory."""
        look_dir = self._look_dir(name)
        look_dir.mkdir(parents=True, exist_ok=True)

        self._to_rgba_image(layer).save(look_dir / _LAYER_FILE)

        if isinstance(meta, LookMeta):
            meta_obj = meta
        else:
            meta_obj = LookMeta(name=name, **(meta or {}))
        meta_obj.name = name
        (look_dir / _META_FILE).write_text(
            json.dumps(meta_obj.to_dict(), indent=2), encoding="utf-8"
        )
        return look_dir

    def load_layer(self, name: str) -> np.ndarray:
        """Load a makeup layer as an RGBA uint8 array (H, W, 4)."""
        path = self._look_dir(name) / _LAYER_FILE
        if not path.exists():
            raise FileNotFoundError(f"No layer for look '{name}' at {path}")
        return np.asarray(Image.open(path).convert("RGBA"))

    def load_meta(self, name: str) -> LookMeta:
        path = self._look_dir(name) / _META_FILE
        if not path.exists():
            return LookMeta(name=name)
        return LookMeta.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_looks(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and (p / _LAYER_FILE).exists()
        )

    def delete(self, name: str) -> None:
        shutil.rmtree(self._look_dir(name), ignore_errors=True)

    def export_zip(self, name: str, out_path: str | Path) -> Path:
        """'Share' a look by zipping its folder. Placeholder for a real share link."""
        look_dir = self._look_dir(name)
        if not look_dir.exists():
            raise FileNotFoundError(f"No look named '{name}'.")
        out_path = Path(out_path)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in look_dir.iterdir():
                zf.write(f, arcname=f.name)
        return out_path
