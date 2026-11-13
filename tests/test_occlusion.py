"""Occlusion masking: the segmenter mask itself, plus the gates it feeds in extract/apply."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from makeup_transfer import occlusion
from makeup_transfer.apply import apply_look
from makeup_transfer.extract import extract_look, repair_occluded_alpha, skin_gate_uv
from makeup_transfer.preprocess import detect_face
from makeup_transfer.storage import LookStore
from makeup_transfer.uv_unwrap import mirror_fill_pair, region_masks_uv

DATA = Path("data/ffhq_makeup")
MAKEUP = DATA / "002386_makeup_03.jpg"   # hair strand across the forehead
BARE = DATA / "002386_bare.jpg"
GLASSES = DATA / "003417_bare.jpg"       # heavy spectacle frames over brows/eyes

if not MAKEUP.exists():
    pytest.skip("FFHQ-Makeup samples not present", allow_module_level=True)

needs_model = pytest.mark.skipif(
    not occlusion.available(), reason="segmenter weight not present (see README setup)"
)


# --- the pure gate helpers (no model needed) ------------------------------------------------
def test_gates_are_no_ops_without_a_mask():
    """Every occlusion hook must degrade to "do nothing" when the model is unavailable."""
    alpha = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    assert skin_gate_uv(None, np.ones((64, 64), np.float32)) is None
    assert np.array_equal(repair_occluded_alpha(alpha, None), alpha)


def test_repair_takes_alpha_from_the_visible_mirrored_half():
    alpha = np.zeros((8, 8), np.float32)
    alpha[:, :4] = 0.9   # junk read off an occluder
    alpha[:, 4:] = 0.2   # the real matte on the visible half
    skin = np.zeros((8, 8), np.float32)
    skin[:, 4:] = 1.0

    out = repair_occluded_alpha(alpha, skin)
    assert np.allclose(out[:, 4:], 0.2)          # visible half untouched
    assert np.allclose(out[:, :4], 0.2)          # occluded half repaired from its mirror


def test_repair_zeroes_alpha_where_both_halves_are_occluded():
    alpha = np.full((8, 8), 0.9, np.float32)
    out = repair_occluded_alpha(alpha, np.zeros((8, 8), np.float32))
    assert np.allclose(out, 0.0)


def test_mirror_fill_pair_treats_both_shots_identically():
    """The whole point: a pixel is either kept in both shots or mirrored in both."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    vis_a = np.ones((16, 16), np.float32)
    vis_b = np.ones((16, 16), np.float32)
    vis_a[:, :8] = 0.0  # occluded in shot A only

    fa, fb, vis = mirror_fill_pair(a, b, vis_a, vis_b)
    # Left half: both taken from the mirrored right half. Right half: both kept as-is.
    assert np.array_equal(fa[:, :8], a[:, ::-1][:, :8])
    assert np.array_equal(fb[:, :8], b[:, ::-1][:, :8])
    assert np.array_equal(fb[:, 8:], b[:, 8:])
    assert vis[:, 8:].min() == 1.0


# --- the real model ---------------------------------------------------------------------------
@needs_model
def test_mask_keeps_the_face_and_rejects_the_hair():
    face = detect_face(MAKEUP)
    skin = face.skin_visibility
    assert skin is not None and skin.shape == face.image.shape[:2]
    assert skin.min() >= 0.0 and skin.max() <= 1.0

    # Cheeks are skin; the region above the head is not.
    cheeks = face.masks["face"] > 0
    assert skin[cheeks].mean() > 0.8
    assert skin[:20, :].mean() < 0.2

    # Brows and lips must survive — they are makeup regions, not occluders.
    for name in ("left_brow", "right_brow", "lips"):
        assert skin[face.masks[name] > 0].mean() > 0.7, name


@needs_model
def test_mask_is_cached_on_the_face():
    face = detect_face(MAKEUP)
    assert face.skin_visibility is face.skin_visibility


@needs_model
def test_extraction_replaces_hair_with_skin_in_the_uv_texture(tmp_path):
    """The strand across this subject's forehead must not survive into the look."""
    store = LookStore(root=tmp_path / "looks")
    _, _, raw = extract_look(MAKEUP, BARE, "raw", store=store, occlusion=False)
    _, _, masked = extract_look(MAKEUP, BARE, "masked", store=store, occlusion=True)

    # The chart hull overhangs the triangulated face slightly; drop the unrendered black margin.
    rendered = (region_masks_uv(masked["alpha"].shape[0])["face"] > 0) & (
        masked["uv_with"].max(axis=2) > 0)
    skin_pair = masked["skin_pair"]
    # Covered on this side, clean on the mirrored one: exactly the case the fill must repair.
    hair = rendered & (skin_pair < 0.05) & (skin_pair[:, ::-1] > 0.95)
    assert hair.sum() > 100, "expected some one-sided occlusion on this pair"

    mirrored = masked["uv_with"][:, ::-1]
    assert np.abs(masked["uv_with"][hair].astype(int) - mirrored[hair].astype(int)).max() <= 1
    # ...and that is a real change: the un-masked run blended in the hair it saw.
    changed = np.abs(masked["uv_with"][hair].astype(int) - raw["uv_with"][hair].astype(int))
    assert changed.mean() > 2


@needs_model
def test_apply_leaves_occluders_untouched(tmp_path):
    """Glasses frames crossing the brows must not pick up brow/eye color."""
    store = LookStore(root=tmp_path / "looks")
    extract_look(MAKEUP, BARE, "look", store=store)

    target = cv2.imread(str(GLASSES)).astype(np.float32)
    skin = detect_face(GLASSES).skin_visibility
    strengths = {"lips": 1.0, "eyes": 1.2, "brows": 1.2, "skin": 1.0}

    changes = {}
    for occ in (False, True):
        result, dbg = apply_look(
            "look", GLASSES, store=store, strengths=strengths, occlusion=occ,
        )
        delta = np.abs(result.astype(np.float32) - target).mean(axis=2)
        chart = dbg["alpha_img"] > 0
        changes[occ] = (delta[chart & (skin < 0.5)].mean(), delta[chart & (skin >= 0.5)].mean())

    on_occluder_off, on_skin_off = changes[False]
    on_occluder_on, on_skin_on = changes[True]
    assert on_occluder_off > 1.0, "test target no longer has makeup landing on its glasses"
    assert on_occluder_on < 0.25 * on_occluder_off  # occluders are left essentially alone
    assert on_skin_on > 0.9 * on_skin_off           # while the skin still gets its makeup
