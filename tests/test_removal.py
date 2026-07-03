"""Real-inference tests for the SSAT bare-face estimator (GPU, a few seconds each)."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
from makeup_transfer import removal  # noqa: E402

if not (removal.SSAT_WEIGHTS.exists() and removal.PARSING_WEIGHTS.exists()):
    pytest.skip("third-party weights not vendored (see Task 1)", allow_module_level=True)

import cv2  # noqa: E402
from makeup_transfer.preprocess import detect_face  # noqa: E402

DATA = Path("data/ffhq_makeup")
MAKEUP = DATA / "002386_makeup_03.jpg"
TRUE_BARE = DATA / "002386_bare.jpg"
REFERENCE = DATA / "003417_bare.jpg"  # different identity: no cheating


@pytest.fixture(scope="module")
def pseudo_bare():
    est = removal.BareFaceEstimator(default_reference=REFERENCE)
    return est.estimate_bare(MAKEUP)


def test_output_matches_input_frame(pseudo_bare):
    src = cv2.imread(str(MAKEUP))
    assert pseudo_bare.shape == src.shape
    assert pseudo_bare.dtype == np.uint8


def test_face_still_detectable(tmp_path, pseudo_bare):
    p = tmp_path / "pseudo_bare.png"
    cv2.imwrite(str(p), pseudo_bare)
    face = detect_face(p)
    assert face.landmarks.shape == (468, 2)


def test_lips_move_toward_true_bare(pseudo_bare):
    """Removal must pull lip chroma toward the ground-truth bare face.

    FFHQ-Makeup pairs share the FFHQ crop, so indexing the bare image with the
    makeup photo's lip mask is aligned enough for a region-MEAN comparison.
    """
    from skimage import color

    def lab(bgr):
        return color.rgb2lab(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0)

    face_m = detect_face(MAKEUP)
    lips = face_m.masks["lips"] > 0
    bare_mean = lab(cv2.imread(str(TRUE_BARE)))[lips].mean(axis=0)
    d_makeup = np.abs(lab(face_m.image)[lips].mean(axis=0) - bare_mean)[1:].sum()
    d_pseudo = np.abs(lab(pseudo_bare)[lips].mean(axis=0) - bare_mean)[1:].sum()
    assert d_pseudo < d_makeup, f"removal made lips LESS bare: {d_pseudo:.2f} >= {d_makeup:.2f}"
