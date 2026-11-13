"""Diffusion engine: the pieces that need no weights (structure drawing, portrait framing, the
backend plumbing and network-output cache), plus an opt-in end-to-end run.

The real model is ~7 GB and takes a minute per image, so the end-to-end test only runs with
``MAKEUP_TEST_DIFFUSION=1`` and the weights installed.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from makeup_transfer import diffusion, engine
from makeup_transfer.preprocess import FaceData

DATA = Path("data/ffhq_makeup")
TARGET = DATA / "003125_bare.jpg"
REFERENCE = DATA / "002386_makeup_03.jpg"


def test_landmark_map_is_68_distinct_mesh_points():
    assert len(diffusion.MP_TO_68) == 68
    assert len(set(diffusion.MP_TO_68)) == 68
    assert all(0 <= i < 468 for i in diffusion.MP_TO_68)


def test_structure_image_draws_the_trained_colours():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    pts = np.zeros((68, 2), np.float32)
    pts[:17] = np.stack([np.linspace(100, 400, 17), 350 + 60 * np.sin(np.linspace(0, np.pi, 17))], 1)
    pts[17:36] = rng.uniform(150, 350, (19, 2))
    pts[36:42] = [[170, 200], [185, 190], [205, 190], [220, 200], [205, 210], [185, 210]]
    pts[42:48] = pts[36:42] + [120, 0]
    pts[48:60] = [[200 + 60 * np.cos(a), 330 + 25 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    pts[60:68] = [[200 + 35 * np.cos(a), 330 + 10 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    img = diffusion.structure_image(pts, 512)
    assert img.shape == (512, 512, 3) and img.dtype == np.uint8
    assert tuple(img[0, 0]) == (0, 0, 0)                       # black background
    assert tuple(img[200, 195]) == (255, 0, 255)               # filled eye: magenta
    assert tuple(img[330, 200]) == (0, 0, 255)                 # inner lips over outer: blue
    assert tuple(img[330, 250]) == (0, 255, 255)               # outer lips ring: cyan
    assert (img == (0, 255, 0)).all(axis=2).sum() > 200        # jaw line: lime


def _face(h=600, w=400):
    rng = np.random.default_rng(1)
    pts = np.stack([rng.uniform(150, 250, 468), rng.uniform(250, 380, 468)], 1).astype(np.float32)
    return FaceData(image=np.zeros((h, w, 3), np.uint8), landmarks=pts, masks={})


def test_portrait_crop_is_looser_than_tight_and_inside_the_image():
    face = _face()
    tight, portrait = engine.face_crop(face), engine.face_crop(face, "portrait")
    assert portrait.size > tight.size
    for c in (tight, portrait):
        assert 0 <= c.x0 and c.x0 + c.size <= 400
        assert 0 <= c.y0 and c.y0 + c.size <= 600


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        engine.get_backend("nope")
    ok, why = engine.backend_available("nope")
    assert not ok and why


class _FakeEngine:
    """Stands in for a slow backend: counts calls, returns a brightened input."""
    name, size, framing = "fake-diffusion", 64, "portrait"

    def __init__(self):
        self.calls = 0
        self.seen = {}

    def transfer_crop(self, target_rgb, reference_rgb, *, landmarks=None, progress=None, **opts):
        self.calls += 1
        self.seen = {"landmarks": landmarks, **opts}
        if progress:
            progress(1, 1)
        return np.clip(target_rgb.astype(int) + 40, 0, 255).astype(np.uint8)


needs_parser = pytest.mark.skipif(not engine.BISENET_WEIGHTS.exists() or not TARGET.exists(),
                                  reason="BiSeNet weights / FFHQ samples not present")


@needs_parser
def test_slow_backend_output_is_cached_across_slider_changes():
    fake = _FakeEngine()
    steps = []
    a = engine.transfer_makeup(TARGET, REFERENCE, engine=fake, seed=7, guidance=1.6,
                               progress=lambda i, n: steps.append((i, n)))
    assert fake.calls == 1 and steps == [(1, 1)]
    assert fake.seen["seed"] == 7 and fake.seen["guidance"] == 1.6
    assert fake.seen["landmarks"].shape == (468, 2)
    assert a.backend == "fake-diffusion" and a.net_out.shape == (64, 64, 3)
    # same inputs, different strengths/complexion: composite only, no second network call
    b = engine.transfer_makeup(TARGET, REFERENCE, engine=fake, seed=7, guidance=1.6,
                               strengths={"lips": 0.0}, complexion=1.0)
    assert fake.calls == 1
    assert not np.array_equal(a.image, b.image)
    # a different seed or guidance is a different generation
    engine.transfer_makeup(TARGET, REFERENCE, engine=fake, seed=8, guidance=1.6)
    engine.transfer_makeup(TARGET, REFERENCE, engine=fake, seed=7, guidance=2.0)
    engine.transfer_makeup(TARGET, REFERENCE, engine=fake, seed=7, guidance=1.6, strength=0.5)
    assert fake.calls == 4
    assert fake.seen["strength"] == 0.5


@needs_parser
def test_remove_glasses_leaves_a_bare_faced_reference_alone():
    ref = engine._reference_crop_rgb(REFERENCE, 256, "portrait")
    assert np.array_equal(engine.remove_glasses(ref), ref)


@needs_parser
def test_remove_glasses_paints_out_frames_but_keeps_the_eyes():
    glasses = DATA / "003417_makeup_05.jpg"
    ref = engine._reference_crop_rgb(glasses, 512, "portrait")
    clean = engine.remove_glasses(ref)
    changed = np.abs(clean.astype(int) - ref.astype(int)).sum(axis=2) > 30
    assert changed.mean() > 0.002                         # frames were painted out
    from makeup_transfer.preprocess import detect_face_bgr
    import cv2
    face = detect_face_bgr(cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    eyes = (face.masks["left_eye"] > 0) | (face.masks["right_eye"] > 0)
    assert not changed[eyes].any()                        # eye openings untouched


@pytest.mark.skipif(
    not (os.environ.get("MAKEUP_TEST_DIFFUSION") and diffusion.available() and TARGET.exists()),
    reason="set MAKEUP_TEST_DIFFUSION=1 with the diffusion weights installed (slow)",
)
def test_diffusion_end_to_end():
    res = engine.transfer_makeup(TARGET, REFERENCE, backend="diffusion", steps=8)
    assert res.backend == "stable-makeup"
    assert res.net_out.shape == (512, 512, 3)
    assert res.net_out.std() > 20                      # an image, not a blank/NaN frame
    assert not np.array_equal(res.image, res.target)
