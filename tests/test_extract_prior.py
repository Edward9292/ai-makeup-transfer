"""Integration: prior-based extraction on a real FFHQ pair, against a tiny fixture prior."""
from pathlib import Path

import numpy as np
import pytest

from makeup_transfer.extract import extract_look, unwrap_to_uv
from makeup_transfer.priors import build_prior_from_diffs, lab_diff_uv
from makeup_transfer.storage import LookStore
from makeup_transfer.uv_unwrap import region_masks_uv

DATA = Path("data/ffhq_makeup")
MAKEUP = DATA / "002386_makeup_03.jpg"
BARE = DATA / "002386_bare.jpg"

if not MAKEUP.exists():
    pytest.skip("FFHQ-Makeup samples not present", allow_module_level=True)


@pytest.fixture(scope="module")
def tiny_prior():
    """A 3-component prior from the sample pairs. Not a real basis — just enough to run."""
    diffs = []
    _, uv_b, _ = unwrap_to_uv(BARE)
    for name in ["002386_makeup_01.jpg", "002386_makeup_03.jpg", "002386_makeup_05.jpg"]:
        _, uv_m, _ = unwrap_to_uv(DATA / name)
        diffs.append(lab_diff_uv(uv_m, uv_b))
    return build_prior_from_diffs(np.stack(diffs), max_components=3)


def test_extract_look_with_prior_puts_alpha_on_the_lips(tmp_path, tiny_prior):
    store = LookStore(root=tmp_path / "looks")
    layer, skin_lab, debug = extract_look(
        MAKEUP, BARE, "prior-test", store=store, prior=tiny_prior,
    )

    assert layer.shape[2] == 4
    alpha = debug["alpha"]
    masks = region_masks_uv(alpha.shape[0])
    lips = masks["lips"] > 0
    face = masks["face"] > 0

    # Lipstick is the strongest signal in this look: lip alpha must clearly exceed face-average.
    assert alpha[lips].mean() > 2.0 * alpha[face].mean()
    assert len(skin_lab) == 3
    assert store.load_meta("prior-test").blend_params["skin_lab_mean"]


def test_prior_none_is_byte_for_byte_the_baseline(tmp_path):
    """The invariant that protects every existing look: no prior => today's extractor, exactly.

    Compares extract_look(prior=None) against extract_layer called directly on the same UV pair.
    """
    from makeup_transfer.extract import extract_layer, skin_gate_uv
    from makeup_transfer.occlusion import skin_visibility_uv
    from makeup_transfer.uv_unwrap import mirror_fill_pair

    store = LookStore(root=tmp_path / "looks")
    layer, _, _ = extract_look(MAKEUP, BARE, "baseline", store=store)

    # Same pair path extract_look takes: raw unwrap, joint mirror-fill, pair skin gate.
    face_m, uv_m, vis_m = unwrap_to_uv(MAKEUP, mirror=False)
    face_b, uv_b, vis_b = unwrap_to_uv(BARE, mirror=False)
    uv_m, uv_b, vis = mirror_fill_pair(uv_m, uv_b, vis_m, vis_b)
    skin_pair = skin_gate_uv(skin_visibility_uv(face_m, uv_m.shape[0]),
                             skin_visibility_uv(face_b, uv_b.shape[0]))
    expected, _, _ = extract_layer(uv_m, vis, uv_b, vis, sym=0.5, skin_pair=skin_pair)

    np.testing.assert_array_equal(layer, expected)


def test_prior_path_is_recorded_in_the_meta(tmp_path, tiny_prior):
    """`extract_method` must survive the storage round-trip (it rides inside blend_params)."""
    store = LookStore(root=tmp_path / "looks")
    extract_look(MAKEUP, BARE, "m-prior", store=store, prior=tiny_prior)
    extract_look(MAKEUP, BARE, "m-base", store=store)

    assert store.load_meta("m-prior").blend_params["extract_method"] == "prior"
    assert store.load_meta("m-base").blend_params["extract_method"] == "difference"
