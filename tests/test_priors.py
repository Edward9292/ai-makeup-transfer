"""Prior model: container, fit, nuisance rejection, residual. All synthetic — no data, no GPU."""
import numpy as np
import pytest

from makeup_transfer.priors import MakeupPrior, PRIOR_SIZE, face_mask_flat


def test_face_mask_flat_shape():
    m = face_mask_flat(PRIOR_SIZE)
    assert m.shape == (PRIOR_SIZE * PRIOR_SIZE,)
    assert m.dtype == bool
    assert 0 < m.sum() < m.size          # a real subset of the chart


def test_prior_save_load_roundtrip(tmp_path):
    d = 3 * int(face_mask_flat(PRIOR_SIZE).sum())
    rng = np.random.default_rng(0)
    prior = MakeupPrior(
        mean=rng.normal(size=d).astype(np.float32),
        components=rng.normal(size=(4, d)).astype(np.float32),
        variances=np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32),
        size=PRIOR_SIZE,
    )
    path = tmp_path / "prior.npz"
    prior.save(path)
    back = MakeupPrior.load(path)

    assert back.size == PRIOR_SIZE
    assert back.n_components == 4
    np.testing.assert_allclose(back.mean, prior.mean)
    np.testing.assert_allclose(back.components, prior.components)
    np.testing.assert_allclose(back.variances, prior.variances)


def test_load_missing_file_names_the_build_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="--build"):
        MakeupPrior.load(tmp_path / "nope.npz")


from makeup_transfer.priors import fit, nuisance_basis


def _toy_prior(size=PRIOR_SIZE, k=6, seed=0):
    """A prior whose components are smooth-ish random fields — stands in for real makeup.

    NOTE: iid Gaussian vectors in R^d are near-orthogonal to the 18-dim smooth nuisance subspace
    by construction (measured overlap ~3e-4 at PRIOR_SIZE). That makes this prior USELESS for
    testing nuisance rejection — the fit would separate makeup from nuisance even if the nuisance
    basis were deleted entirely, because random components barely alias with `nuisance_basis`
    output to begin with. Use `_collinear_prior` for anything that claims to test rejection.
    """
    d = 3 * int(face_mask_flat(size).sum())
    rng = np.random.default_rng(seed)
    comps = rng.normal(size=(k, d)).astype(np.float32)
    comps /= np.linalg.norm(comps, axis=1, keepdims=True)
    return MakeupPrior(
        mean=np.zeros(d, dtype=np.float32),
        components=comps,
        variances=np.linspace(4.0, 1.0, k).astype(np.float32),
        size=size,
    )


def _collinear_prior(size=PRIOR_SIZE, seed=1):
    """A prior with two components deliberately collinear with the nuisance subspace.

    Real foundation makeup is close to a uniform Lab offset, and real shading is a smooth
    gradient — both lie inside (or very near) the span of `nuisance_basis`. That's the realistic,
    adversarial case for nuisance rejection: unlike `_toy_prior`'s random fields, these components
    genuinely alias with the nuisance model, so the fit's ability to reject drift as non-makeup
    has to come from the actual mechanism (comparing how well nuisance vs. makeup explain a
    *smooth* residual), not from lucky near-orthogonality.

    components[0]: a uniform offset confined to the L channel block — exactly the shape of the
        nuisance basis's per-channel constant term for L.
    components[1]: a smooth left-right ramp repeated identically across all three channel blocks —
        a linear combination of each channel's "x" nuisance column.
    components[2:4]: ordinary random fields, so K=4 like a small real basis.
    """
    mask = face_mask_flat(size)
    p = int(mask.sum())
    d = 3 * p

    c0 = np.zeros(d, dtype=np.float32)
    c0[:p] = 1.0
    c0 /= np.linalg.norm(c0)

    yy, xx = np.mgrid[0:size, 0:size]
    xn = ((xx / (size - 1)) * 2 - 1).reshape(-1)[mask].astype(np.float32)
    c1 = np.tile(xn, 3)
    c1 /= np.linalg.norm(c1)

    rng = np.random.default_rng(seed)
    extra = rng.normal(size=(2, d)).astype(np.float32)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)

    comps = np.stack([c0, c1, extra[0], extra[1]], axis=0).astype(np.float32)
    return MakeupPrior(
        mean=np.zeros(d, dtype=np.float32),
        components=comps,
        variances=np.linspace(4.0, 1.0, 4).astype(np.float32),
        size=size,
    )


def test_nuisance_basis_spans_constant_and_gradient():
    n = nuisance_basis(PRIOR_SIZE)
    d = 3 * int(face_mask_flat(PRIOR_SIZE).sum())
    assert n.shape == (d, 18)
    # A per-channel constant offset must be exactly representable.
    offset = np.zeros(d, dtype=np.float32)
    third = d // 3
    offset[:third] = 2.5                      # L channel shifted
    coef, *_ = np.linalg.lstsq(n, offset, rcond=None)
    np.testing.assert_allclose(n @ coef, offset, atol=1e-4)


def test_basis_roundtrip_recovers_known_component():
    """A difference synthesized as 2x one basis component recovers c ~= [0, 2, 0, 0, 0, 0].

    Asserting on the reconstructed field (`res.makeup`) is too weak to be a real test here: in
    R^31998 a single unit component has max|truth| ~= 0.05, so an atol=0.05 comparison against
    the reconstructed field would pass even if `fit` returned all-zero makeup. Assert on the
    recovered coefficients instead, which is what the fit actually solves for.
    """
    prior = _toy_prior()
    truth = 2.0 * prior.components[1]
    res = fit(prior, truth.astype(np.float32), lam=1e-6)
    expected_c = np.zeros(prior.n_components, dtype=np.float32)
    expected_c[1] = 2.0
    np.testing.assert_allclose(res.c, expected_c, atol=0.01)


def test_nuisance_rejection_is_the_003417_fix():
    """ZERO makeup + a global colour offset + a smooth gradient => recovered makeup ~= 0.

    This is exactly the SSAT skin-tone-drift failure. A raw chromaticity difference reports the
    whole face as makeup here; the prior fit must not.

    Uses `_collinear_prior`, NOT `_toy_prior`: `_toy_prior`'s iid Gaussian components are
    near-orthogonal to the nuisance subspace by construction, so this test would pass with a huge
    margin even if the nuisance basis were deleted from `fit` entirely — it would prove nothing
    about the actual rejection mechanism. `_collinear_prior`'s first two components are
    deliberately collinear with (or a linear combination of) nuisance basis columns — the
    realistic case, since foundation is close to a uniform Lab offset. If the mechanism that
    separates makeup from nuisance by *smoothness* were broken, this drift (itself smooth) would
    leak into `res.makeup` through those collinear components.
    """
    prior = _collinear_prior()
    size = PRIOR_SIZE
    mask = face_mask_flat(size)
    yy, xx = np.mgrid[0:size, 0:size]
    xn = ((xx / (size - 1)) * 2 - 1).reshape(-1)[mask]
    yn = ((yy / (size - 1)) * 2 - 1).reshape(-1)[mask]

    # per-channel constant drift + a smooth left-right shading gradient — no makeup at all
    drift = np.concatenate([
        1.8 + 0.9 * xn,          # L
        2.4 + 0.6 * yn,          # a
        -1.5 + 0.4 * xn,         # b
    ]).astype(np.float32)

    res = fit(prior, drift, lam=1.0)

    assert np.abs(res.makeup).mean() < 0.05 * np.abs(drift).mean(), (
        "prior fit absorbed skin-tone drift as makeup — the 003417 bug"
    )
    np.testing.assert_allclose(res.nuisance, drift, atol=0.1)


def test_ridge_does_not_leak_onto_nuisance_coefficients():
    """A huge ridge on the makeup coefficients must not damp the (unpenalized) nuisance fit.

    `fit` builds a (K+18) penalty diagonal that is zero over the nuisance block. If that zero
    padding were ever wrong — the penalty applied to the whole coefficient vector instead of just
    `c` — a large `lam` would shrink `k` too, and the nuisance recovery below would collapse
    toward zero even though nothing is supposed to be penalizing it. `lam=1.0` can't catch this:
    the nuisance Gram diagonal is ~1e4, so a leaked `lam=1.0` penalty is invisible next to it.
    `lam=1e6` is not.
    """
    prior = _toy_prior()
    size = PRIOR_SIZE
    mask = face_mask_flat(size)
    yy, xx = np.mgrid[0:size, 0:size]
    xn = ((xx / (size - 1)) * 2 - 1).reshape(-1)[mask]
    yn = ((yy / (size - 1)) * 2 - 1).reshape(-1)[mask]

    drift = np.concatenate([
        1.8 + 0.9 * xn,          # L
        2.4 + 0.6 * yn,          # a
        -1.5 + 0.4 * xn,         # b
    ]).astype(np.float32)

    res = fit(prior, drift, lam=1e6)
    np.testing.assert_allclose(res.nuisance, drift, atol=0.1)


def test_lab_diff_uv_layout_is_blocked_not_interleaved():
    """Two uniform-colour synthetic BGR textures differing by a known amount: the returned vector
    must be [L block][a block][b block] concatenated, not interleaved.

    This layout is a contract a later task must invert exactly (to turn a flat makeup vector back
    into a Lab UV texture). A transposed or interleaved layout would sail through the rest of the
    suite — everything else only checks aggregate statistics — and would only surface downstream
    as a garbled makeup layer.
    """
    from skimage import color

    from makeup_transfer.priors import lab_diff_uv

    size = 32
    src_size = 64  # exercise the internal cv2.resize path, not just an identity resize
    color_with_bgr = np.array([80, 120, 200], dtype=np.uint8)
    color_without_bgr = np.array([90, 110, 180], dtype=np.uint8)

    uv_with = np.tile(color_with_bgr, (src_size, src_size, 1))
    uv_without = np.tile(color_without_bgr, (src_size, src_size, 1))

    d = lab_diff_uv(uv_with, uv_without, size=size)

    mask_count = int(face_mask_flat(size).sum())
    assert d.shape == (3 * mask_count,)

    lab_with = color.rgb2lab(color_with_bgr[::-1].reshape(1, 1, 3) / 255.0).reshape(3)
    lab_without = color.rgb2lab(color_without_bgr[::-1].reshape(1, 1, 3) / 255.0).reshape(3)
    expected = (lab_with - lab_without).astype(np.float32)

    l_block = d[:mask_count]
    a_block = d[mask_count:2 * mask_count]
    b_block = d[2 * mask_count:]

    np.testing.assert_allclose(l_block, np.full(mask_count, expected[0]), atol=1e-3)
    np.testing.assert_allclose(a_block, np.full(mask_count, expected[1]), atol=1e-3)
    np.testing.assert_allclose(b_block, np.full(mask_count, expected[2]), atol=1e-3)
