"""End-to-end single-photo extraction (GPU + weights; skips if not vendored)."""
from pathlib import Path

import pytest

pytest.importorskip("torch")
from makeup_transfer import removal  # noqa: E402

if not (removal.SSAT_WEIGHTS.exists() and removal.PARSING_WEIGHTS.exists()):
    pytest.skip("third-party weights not vendored (see Task 1)", allow_module_level=True)

from makeup_transfer.extract import extract_look_single  # noqa: E402
from makeup_transfer.storage import LookStore  # noqa: E402

MAKEUP = Path("data/ffhq_makeup/003417_makeup_05.jpg")
REFERENCE = Path("data/ffhq_makeup/002386_bare.jpg")


def test_extract_look_single(tmp_path):
    store = LookStore(root=tmp_path / "looks")
    layer, skin_lab, debug = extract_look_single(
        MAKEUP, "single-test", store=store,
        reference=REFERENCE, pseudo_out=tmp_path / "pseudo.png",
    )
    assert layer.shape[2] == 4 and layer[..., 3].max() > 0
    assert (tmp_path / "pseudo.png").exists()
    meta = store.load_meta("single-test")
    assert meta.source_mode == "single"
    assert len(skin_lab) == 3
