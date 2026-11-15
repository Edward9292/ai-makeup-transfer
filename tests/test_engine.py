"""Reference-based makeup engine: crop geometry, masks and complexion damping (pure numpy),
plus an end-to-end transfer + look round-trip when the weights are present."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_transfer import engine
from makeup_transfer.storage import LookStore

DATA = Path("data/ffhq_makeup")
TARGET = DATA / "003125_bare.jpg"
REFERENCE = DATA / "002386_makeup_03.jpg"   # red lip, soft shadow
GLASSES_TARGET = DATA / "003417_bare.jpg"   # frames over brows/eyes

needs_weights = pytest.mark.skipif(
    not engine.available(), reason="engine weights not present (python -m makeup_transfer.fetch_weights)"
)
needs_data = pytest.mark.skipif(not TARGET.exists(), reason="FFHQ-Makeup samples not present")


def _parse_fixture(h: int = 128, w: int = 128) -> np.ndarray:
    """A cartoon parse map: skin face oval with lips, brows, eyes, hair on top, bg around."""
    p = np.zeros((h, w), np.uint8)
    cv2.ellipse(p, (w // 2, h // 2), (w // 3, h // 2 - 8), 0, 0, 360, engine._SKIN, -1)
    p[:20] = engine._HAIR
    cv2.rectangle(p, (40, 40), (54, 46), engine._L_EYE, -1)
    cv2.rectangle(p, (74, 40), (88, 46), engine._R_EYE, -1)
    cv2.rectangle(p, (38, 30), (56, 34), engine._L_BROW, -1)
    cv2.rectangle(p, (72, 30), (90, 34), engine._R_BROW, -1)
    cv2.rectangle(p, (52, 84), (76, 90), engine._U_LIP, -1)
    cv2.rectangle(p, (52, 91), (76, 98), engine._L_LIP, -1)
    return p


# --- pure helpers --------------------------------------------------------------------------
def test_region_weights_partition_the_face():
    p = _parse_fixture()
    ones = engine.region_weights(p, {"lips": 1, "eyes": 1, "brows": 1, "skin": 1})
    face = np.isin(p, [engine._SKIN, engine._U_LIP, engine._L_LIP, engine._L_BROW,
                       engine._R_BROW, engine._L_EYE, engine._R_EYE])
    assert np.allclose(ones[face], 1.0, atol=1e-4)      # strengths of 1 everywhere -> 1
    lips_off = engine.region_weights(p, {"lips": 0.0})
    assert lips_off[94, 64] < 0.15                     # inside the (feathered) lower lip
    assert lips_off[60, 64] > 0.95                     # on the cheek, untouched


def test_edit_mask_excludes_hair_eyes_and_background():
    m = engine.edit_mask(_parse_fixture())
    assert m[60, 64] > 0.95          # skin
    assert m[5, 64] < 0.05           # hair
    assert m[64, 2] < 0.05           # background
    assert m[43, 47] < 0.6           # inside the eye opening (feathered, but mostly out)


def test_complexion_keep_one_is_identity():
    p = _parse_fixture()
    base = np.full((128, 128, 3), 150, np.uint8)
    delta = np.random.default_rng(0).normal(0, 10, (128, 128, 3)).astype(np.float32)
    out = engine._damp_complexion(base, delta, p, keep=1.0)
    assert np.array_equal(out, delta)


def test_complexion_zero_removes_a_uniform_lift_but_keeps_warm_blush():
    p = _parse_fixture()
    base = np.full((128, 128, 3), 140, np.uint8)
    delta = np.full((128, 128, 3), 30.0, np.float32)           # uniform lightening everywhere
    out = engine._damp_complexion(base, delta, p, keep=0.0)
    assert abs(out[60, 64]).max() < 8.0                        # skin: the lift is (nearly) gone
    assert abs(out[94, 64] - 30.0).max() < 8.0                 # lips: kept (feathered edge)
    # a warm (red) local patch on the cheek survives with keep=0
    delta2 = np.zeros((128, 128, 3), np.float32)
    delta2[55:65, 44:52, 2] = 40.0                             # +R in BGR on the left cheek
    out2 = engine._damp_complexion(base, delta2, p, keep=0.0)
    assert out2[60, 48, 2] > 15.0


def test_face_crop_is_square_and_inside_the_image():
    from makeup_transfer.preprocess import FaceData

    img = np.zeros((300, 200, 3), np.uint8)
    rng = np.random.default_rng(1)
    pts = np.stack([rng.uniform(60, 140, 468), rng.uniform(90, 230, 468)], axis=1).astype(np.float32)
    face = FaceData(image=img, landmarks=pts, masks={})
    c = engine.face_crop(face)
    assert c.size > 0
    assert 0 <= c.x0 and c.x0 + c.size <= 200
    assert 0 <= c.y0 and c.y0 + c.size <= 300


# --- end to end ----------------------------------------------------------------------------
@needs_weights
@needs_data
def test_transfer_changes_lips_and_leaves_background_alone():
    res = engine.transfer_makeup(TARGET, REFERENCE, match_skin=0.0)
    assert res.image.shape == res.target.shape
    ys, xs = res.crop.slice()
    outside = np.ones(res.target.shape[:2], bool)
    outside[ys, xs] = False
    assert np.array_equal(res.image[outside], res.target[outside])
    assert res.body_mask is None
    diff = np.abs(res.image.astype(int) - res.target.astype(int)).mean(axis=2)
    assert diff[ys, xs].mean() > 1.0                       # something was painted
    assert res.net_out.shape == (engine.NET_SIZE, engine.NET_SIZE, 3)


@needs_weights
@needs_data
def test_strength_zero_everywhere_is_a_no_op():
    res = engine.transfer_makeup(
        TARGET, REFERENCE, strengths={"lips": 0, "eyes": 0, "brows": 0, "skin": 0})
    assert np.array_equal(res.image, res.target)


@needs_weights
@needs_data
def test_glasses_on_the_target_stay_clean():
    res = engine.transfer_makeup(GLASSES_TARGET, REFERENCE)
    from makeup_transfer.preprocess import detect_face

    face = detect_face(GLASSES_TARGET)
    skin = face.skin_visibility
    if skin is None:
        pytest.skip("segmenter weight not present")
    occluder = skin < 0.05
    change = np.abs(res.image.astype(int) - res.target.astype(int)).mean(axis=2)
    assert change[occluder].mean() < 0.5


@needs_weights
@needs_data
def test_create_look_then_apply_round_trip(tmp_path):
    from makeup_transfer.apply import apply_look

    store = LookStore(root=tmp_path / "looks")
    ref, look_dir = engine.create_look(REFERENCE, "rose-test", store=store)
    assert (look_dir / "reference.png").is_file()
    assert ref.shape == (engine.REFERENCE_SIZE, engine.REFERENCE_SIZE, 3)
    assert store.list_looks() == ["rose-test"]
    meta = store.load_meta("rose-test")
    assert meta.source_mode == "reference" and meta.engine == "csd-mt"

    result, debug = apply_look("rose-test", TARGET, store=store, strengths={"lips": 1.2})
    assert result.shape == debug["target"].shape
    assert not np.array_equal(result, debug["target"])
    swatches = engine.reference_swatches(ref)
    assert "lips" in swatches and swatches["lips"].startswith("#")


def test_border_fade_is_zero_at_the_edge_and_one_inside():
    f = engine._border_fade(200)
    assert f.shape == (200, 200)
    assert f[0, 100] < 0.1 and f[100, 0] < 0.1 and f[199, 199] < 0.1
    assert f[100, 100] == 1.0 and f[20, 20] == 1.0


# --- skin-tone matching beyond the face ----------------------------------------------------
def _toned(parse: np.ndarray, bgr=(120, 150, 200)) -> np.ndarray:
    img = np.zeros(parse.shape + (3,), np.uint8)
    img[parse != engine._BG] = bgr
    return img


def test_complexion_shift_reads_the_overall_change_and_ignores_local_makeup():
    p = _parse_fixture()
    before = _toned(p)
    full = np.ones(p.shape, np.float32)
    assert engine.complexion_shift(before, before.astype(np.float32), p, full) is None

    after = before.astype(np.float32) * 1.15                     # an even lightening
    gain, da, db = engine.complexion_shift(before, after, p, full)
    assert 1.05 < gain < 1.2
    blush = before.astype(np.float32)
    blush[55:65, 44:52, 2] += 60.0                              # one red patch on a cheek
    assert engine.complexion_shift(before, blush, p, full) is None


def test_match_body_skin_follows_the_mask_and_the_amount():
    img = np.full((40, 60, 3), (120, 150, 200), np.uint8)
    mask = np.zeros((40, 60), np.float32)
    mask[10:30, 20:50] = 1.0
    d = engine.match_body_skin(img, (1.1, 3.0, -2.0), mask)
    assert d.shape == img.shape
    assert np.abs(d[:10]).max() == 0 and np.abs(d[:, :20]).max() == 0   # untouched off the mask
    assert d[20, 30].mean() > 5.0                                      # lighter inside
    half = engine.match_body_skin(img, (1.1, 3.0, -2.0), mask, 0.5)
    assert np.allclose(half, 0.5 * d, atol=1e-3)
    assert np.abs(engine.match_body_skin(img, (1.1, 3.0, -2.0), np.zeros_like(mask))).max() == 0


def test_complexion_field_is_local_near_the_face_and_global_at_the_crop_border():
    p = _parse_fixture(256, 256)
    before = _toned(p)
    after = before.astype(np.float32)
    after[:, 128:] *= 1.2                                       # only the right half lightened
    full = np.ones(p.shape, np.float32)
    f = engine.complexion_field(before, after, p, full, (1.1, 0.0, 0.0))
    assert f.shape == (256, 256, 3)
    assert f[128, 190, 0] > f[128, 66, 0] + 0.1                 # follows the nearby face skin
    assert abs(f[0, 0, 0] - 1.1) < 1e-3                         # the global shift at the border


@needs_weights
@needs_data
def test_matching_carries_the_face_tone_onto_the_neck_only_on_skin():
    off = engine.transfer_makeup(TARGET, REFERENCE, complexion=1.0, match_skin=0.0)
    on = engine.transfer_makeup(TARGET, REFERENCE, complexion=1.0, match_skin=1.0)
    assert on.body_mask is not None and on.body_mask.shape == on.target.shape[:2]
    change = np.abs(on.image.astype(int) - off.image.astype(int)).mean(axis=2)
    assert change[on.body_mask > 0.9].mean() > 0.5              # the neck / ears moved
    assert change[on.body_mask < 0.01].max() <= 1               # nothing else did
    ys, xs = on.crop.slice()
    parse = engine.get_parser().parse(cv2.cvtColor(on.target[ys, xs], cv2.COLOR_BGR2RGB))
    hair = parse == engine._HAIR
    assert change[ys, xs][hair].mean() < 0.3
