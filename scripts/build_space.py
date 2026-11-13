"""Stage the Hugging Face Space folder (CPU demo) into space_build/.

Usage:  python scripts/build_space.py  [dest]

The Space runs the reference-based makeup engine on CPU (CSD-MT + BiSeNet, ~80 MB of
weights, 1-4 s per transfer). Those two ``.pth`` files are copied from ``weights/`` — run
``python -m makeup_transfer.fetch_weights`` first. ``third_party/`` never ships.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "space_assets"

SEED_LOOKS = ["ffhq-002386-03", "ffhq-003125-02", "ffhq-003417-05"]
WEIGHTS = [
    "face_landmarker.task",
    "canonical_face_model.obj",
    "selfie_multiclass_256x256.tflite",  # occlusion masking (hair/hands/glasses), CPU
    "csd_mt.pth",                        # makeup engine (engine.py), CPU
    "bisenet_79999_iter.pth",            # face parsing for the engine, CPU
]

ENTRY_SHIM = '''"""HF Space entrypoint: run the packaged Gradio app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from makeup_transfer.app import LAUNCH_KWARGS, build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)
'''


def stage(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # code (no __pycache__)
    shutil.copytree(ROOT / "src" / "makeup_transfer", dest / "src" / "makeup_transfer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # assets
    (dest / "weights").mkdir()
    for w in WEIGHTS:
        src = ROOT / "weights" / w
        if not src.exists():
            raise SystemExit(f"Missing {src} — run: python -m makeup_transfer.fetch_weights")
        shutil.copy2(src, dest / "weights" / w)
    for look in SEED_LOOKS:
        shutil.copytree(ROOT / "looks" / look, dest / "looks" / look)
    shutil.copytree(ROOT / "data" / "ffhq_makeup", dest / "data" / "ffhq_makeup",
                    ignore=shutil.ignore_patterns("*.zip"))
    # space metadata + entry
    shutil.copy2(ASSETS / "requirements.txt", dest / "requirements.txt")
    shutil.copy2(ASSETS / "README.md", dest / "README.md")
    (dest / "app.py").write_text(ENTRY_SHIM, encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "space_build"
    stage(out)
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"Staged {n} files -> {out}")
