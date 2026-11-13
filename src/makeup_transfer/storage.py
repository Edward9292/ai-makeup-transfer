"""Fake, local-file storage for makeup "looks".

A *look* is the savable/shareable artifact. Since the engine rewrite it is a **reference face**:
a square RGB crop of someone wearing the makeup (``reference.png``), which the transfer network
reads the makeup from at apply time. Older looks hold a UV-space RGBA ``layer.png`` instead and
still load (``apply.py`` falls back to the legacy blend for them).

For now everything lives on disk under ``looks/<name>/``. This module is intentionally a thin
interface so a real backend (object store + DB) can replace it later WITHOUT touching the NN code.

Layout per look::

    looks/<name>/
        reference.png  # square RGB face crop wearing the makeup (engine looks)
        layer.png      # RGBA makeup layer in UV space (legacy looks)
        meta.json      # {name, created, source_with, source_mode, blend_params, notes}

Usage::

    store = LookStore()                      # defaults to <repo>/looks
    store.save_reference("smoky-eye", face_rgb, meta={"source_with": "photo.jpg"})
    ref = store.load_reference("smoky-eye")  # RGB uint8, or None for a legacy look
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
_REFERENCE_FILE = "reference.png"
_META_FILE = "meta.json"


@dataclass
class LookMeta:
    """Metadata stored alongside a makeup layer."""

    name: str
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_with: str | None = None      # filename/path of the with-makeup source
    source_without: str | None = None   # filename/path of the without-makeup source
    blend: str = "multiply"             # how the layer should be composited onto a target
    # Per-region blend recipe used by apply.py, e.g.
    #   {"skin_lab_mean": [L, a, b],
    #    "regions": {"lips": {"mode": "multiply", "strength": 1.0}, ...}}
    # Empty dict -> apply.py falls back to its defaults (keeps old looks loadable).
    blend_params: dict = field(default_factory=dict)
    # "reference" = engine look (reference.png); "pair" / "single" = legacy UV layers.
    source_mode: str = "pair"
    engine: str = ""                    # which transfer engine the look was made for
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
        self._write_meta(look_dir, name, meta)
        return look_dir

    def _write_meta(self, look_dir: Path, name: str, meta: dict | LookMeta | None) -> None:
        meta_obj = meta if isinstance(meta, LookMeta) else LookMeta(name=name, **(meta or {}))
        meta_obj.name = name
        (look_dir / _META_FILE).write_text(
            json.dumps(meta_obj.to_dict(), indent=2), encoding="utf-8"
        )

    def save_reference(
        self,
        name: str,
        reference_rgb: np.ndarray,
        meta: dict | LookMeta | None = None,
    ) -> Path:
        """Persist an engine look: a square RGB face crop wearing the makeup + metadata."""
        arr = np.asarray(reference_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Reference must be RGB (H, W, 3); got shape {arr.shape}.")
        look_dir = self._look_dir(name)
        look_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").save(
            look_dir / _REFERENCE_FILE)
        if isinstance(meta, LookMeta):
            meta.source_mode, meta.engine = "reference", meta.engine or "csd-mt"
        else:
            meta = {"source_mode": "reference", "engine": "csd-mt", **(meta or {})}
        self._write_meta(look_dir, name, meta)
        return look_dir

    def load_reference(self, name: str) -> np.ndarray | None:
        """The look's reference face as RGB uint8 (H, W, 3), or ``None`` for a legacy look."""
        path = self._look_dir(name) / _REFERENCE_FILE
        if not path.exists():
            return None
        return np.asarray(Image.open(path).convert("RGB"))

    def has_reference(self, name: str) -> bool:
        return (self._look_dir(name) / _REFERENCE_FILE).exists()

    def load_layer(self, name: str) -> np.ndarray:
        """Load a legacy makeup layer as an RGBA uint8 array (H, W, 4)."""
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
            if p.is_dir() and ((p / _REFERENCE_FILE).exists() or (p / _LAYER_FILE).exists())
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
