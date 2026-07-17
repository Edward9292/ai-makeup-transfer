"""build_space.stage() must assemble a self-contained CPU Space folder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_space import stage  # noqa: E402


def test_stage_layout(tmp_path):
    dest = tmp_path / "space"
    stage(dest)
    # entry + code
    assert (dest / "app.py").is_file()
    assert (dest / "src" / "makeup_transfer" / "app.py").is_file()
    assert (dest / "src" / "makeup_transfer" / "extract.py").is_file()
    # CPU weights in, GPU weights + third_party out
    assert (dest / "weights" / "face_landmarker.task").is_file()
    assert (dest / "weights" / "canonical_face_model.obj").is_file()
    assert not (dest / "third_party").exists()
    assert not list(dest.rglob("*.pth"))
    # seeds + samples + metadata
    assert (dest / "looks" / "ffhq-002386-03" / "layer.png").is_file()
    assert (dest / "data" / "ffhq_makeup" / "003417_bare.jpg").is_file()
    reqs = (dest / "requirements.txt").read_text()
    assert "torch" not in reqs
    assert "mediapipe" in reqs
    card = (dest / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---")          # HF YAML frontmatter
    assert "sdk: gradio" in card


def test_stage_entry_shim_imports_app(tmp_path):
    dest = tmp_path / "space"
    stage(dest)
    shim = (dest / "app.py").read_text()
    assert "makeup_transfer.app" in shim
