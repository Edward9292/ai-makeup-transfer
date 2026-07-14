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
    """The basis must exactly represent a per-channel constant, a linear gradient, AND a
    degree-2 term — its docstring promises "per-channel constant + degree-2 polynomial in UV
    coords", but until now only the constant term was ever exercised, leaving the x/y/x^2/xy/y^2
    columns completely unchecked.
    """
    n = nuisance_basis(PRIOR_SIZE)
    d = 3 * int(face_mask_flat(PRIOR_SIZE).sum())
    assert n.shape == (d, 18)
    third = d // 3

    # A per-channel constant offset must be exactly representable.
    offset = np.zeros(d, dtype=np.float32)
    offset[:third] = 2.5                      # L channel shifted
    coef, *_ = np.linalg.lstsq(n, offset, rcond=None)
    np.testing.assert_allclose(n @ coef, offset, atol=1e-4)

    # A linear gradient (degree-1 in UV coords) must also be exactly representable.
    mask = face_mask_flat(PRIOR_SIZE)
    yy, xx = np.mgrid[0:PRIOR_SIZE, 0:PRIOR_SIZE]
    x = ((xx / (PRIOR_SIZE - 1)) * 2 - 1).reshape(-1)[mask].astype(np.float32)
    y = ((yy / (PRIOR_SIZE - 1)) * 2 - 1).reshape(-1)[mask].astype(np.float32)
    gradient = np.zeros(d, dtype=np.float32)
    gradient[third:2 * third] = 1.5 * x - 0.7 * y          # a channel
    coef, *_ = np.linalg.lstsq(n, gradient, rcond=None)
    np.testing.assert_allclose(n @ coef, gradient, atol=1e-4)

    # A degree-2 term (curvature + cross term) must also be exactly representable.
    quad = np.zeros(d, dtype=np.float32)
    quad[2 * third:] = 2.0 * x * x - 1.0 * x * y + 0.5 * y * y   # b channel
    coef, *_ = np.linalg.lstsq(n, quad, rcond=None)
    np.testing.assert_allclose(n @ coef, quad, atol=1e-4)


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
    """A spatially non-uniform BGR difference: the returned vector must be [L block][a block]
    [b block] concatenated, not interleaved, AND each block's values must sit at the *correct*
    pixel position within the block.

    Uniform-colour textures (the original version of this test) cannot catch a within-block pixel
    permutation — every pixel in a uniform field has the same value, so a transposed or
    column-major flatten produces byte-for-byte the same output vector as the correct row-major
    flatten. Here the "with" texture carries a horizontal ramp in the Green channel, so the
    Lab difference varies pixel-to-pixel; the expected value at each masked (row, col) is computed
    independently (via `cv2.resize` + `skimage.color.rgb2lab`, not by calling `lab_diff_uv`) and
    matched up using `np.nonzero` on the 2D mask, which walks row-major exactly like
    `.reshape(-1)[mask]` does. A column-major or transposed flatten inside `lab_diff_uv` would
    misalign values against this independently-derived expectation and fail the comparison (verified
    against simulated column-major/transpose bugs before writing this test).
    """
    import cv2
    from skimage import color

    from makeup_transfer.priors import lab_diff_uv

    size = 32
    src_size = 64  # exercise the internal cv2.resize path, not just an identity resize

    yy, xx = np.mgrid[0:src_size, 0:src_size]
    ramp = (xx.astype(np.float32) / (src_size - 1) * 200 + 20)   # 20..220, varies by column only

    uv_with = np.zeros((src_size, src_size, 3), dtype=np.uint8)
    uv_with[..., 0] = 90                       # B, uniform
    uv_with[..., 1] = ramp.astype(np.uint8)    # G, horizontal ramp
    uv_with[..., 2] = 150                      # R, uniform

    uv_without = np.full((src_size, src_size, 3), (90, 110, 150), dtype=np.uint8)

    d = lab_diff_uv(uv_with, uv_without, size=size)

    mask2d = face_mask_flat(size).reshape(size, size)
    mask_count = int(mask2d.sum())
    assert d.shape == (3 * mask_count,)

    # Independent ground truth: same resize/colour-space steps, computed here rather than by
    # calling the function under test.
    small_w = cv2.resize(uv_with, (size, size), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(uv_without, (size, size), interpolation=cv2.INTER_AREA)
    lab_w = color.rgb2lab(cv2.cvtColor(small_w, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    lab_b = color.rgb2lab(cv2.cvtColor(small_b, cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    expected_full = lab_w - lab_b                          # (size, size, 3)

    rows, cols = np.nonzero(mask2d)                        # row-major order, matches reshape(-1)
    expected_l = expected_full[rows, cols, 0]
    expected_a = expected_full[rows, cols, 1]
    expected_b = expected_full[rows, cols, 2]

    l_block = d[:mask_count]
    a_block = d[mask_count:2 * mask_count]
    b_block = d[2 * mask_count:]

    # Sanity: the ramp must actually make these blocks non-uniform, or the test would be no
    # stronger than the uniform-colour version it replaces.
    assert not np.allclose(l_block, l_block[0])
    assert not np.allclose(a_block, a_block[0])
    assert not np.allclose(b_block, b_block[0])

    np.testing.assert_allclose(l_block, expected_l, atol=1e-3)
    np.testing.assert_allclose(a_block, expected_a, atol=1e-3)
    np.testing.assert_allclose(b_block, expected_b, atol=1e-3)


def test_real_makeup_survives_the_ridge_at_default_lam():
    """A plausible, leading-component makeup signal must not be crushed by the production ridge.

    `test_nuisance_rejection_is_the_003417_fix` — and every other test in this file — is
    monotonically *easier* to satisfy the harder `lam` shrinks makeup coefficients toward zero: the
    rejection test only asserts an upper bound on `res.makeup`, and the roundtrip test
    (`test_basis_roundtrip_recovers_known_component`) dodges the ridge entirely with `lam=1e-6`.
    A degenerate `lam` that crushed ALL real makeup to nothing would pass the whole existing suite
    while the extractor emitted an empty makeup layer. This test brackets `lam` from the other
    side, so the two together constrain it from both directions.

    Uses a realistically STEEP variance spectrum (`variances = 100.0 * (0.5 ** arange(K))`,
    geometric decay like a real PCA's eigenvalues) rather than `_toy_prior`'s nearly-flat spectrum
    — the ridge penalty is `lam * mean_variance / variance`, so spectrum shape directly controls how
    hard the leading (highest-variance, most-plausible-makeup) component gets penalized. The
    observed difference is synthesized as a known multiple of that leading component — the
    plausible-makeup regime a real extractor should preserve.

    If this fails at the production default `lam`, DO NOT loosen the threshold: it means the
    default is miscalibrated and is destroying real makeup signal, which must be reported rather
    than masked here (see the Task 3 review-fix report).
    """
    size = PRIOR_SIZE
    d = 3 * int(face_mask_flat(size).sum())
    rng = np.random.default_rng(2)
    k = 6
    comps = rng.normal(size=(k, d)).astype(np.float32)
    comps /= np.linalg.norm(comps, axis=1, keepdims=True)
    prior = MakeupPrior(
        mean=np.zeros(d, dtype=np.float32),
        components=comps,
        variances=(100.0 * (0.5 ** np.arange(k))).astype(np.float32),
        size=size,
    )

    truth_c0 = 3.0
    d_obs = (truth_c0 * prior.components[0]).astype(np.float32)

    res = fit(prior, d_obs)          # exercise fit's actual default lam, not a hardcoded stand-in

    assert res.c[0] > 0.8 * truth_c0, (
        f"the default lam ridge crushed a leading-component makeup coefficient: recovered "
        f"{res.c[0]:.4f} of true {truth_c0} ({100 * res.c[0] / truth_c0:.1f}% retained) — "
        "the production default lam appears miscalibrated; see finding 1 in the Task 3 "
        "review-fix report, do not loosen this threshold to make it pass"
    )


def test_nonzero_prior_mean_round_trips():
    """`makeup = mean + M.c` is a contract a later task inverts; every toy prior elsewhere in this
    suite uses `mean = zeros(D)`, so a sign error or an outright omission of `prior.mean` on either
    side of `fit` (subtracting it from `d_obs`, or adding it back into `res.makeup`) would pass the
    entire existing suite silently. Confirmed independently: deleting both the `- prior.mean` and
    `prior.mean +` lines in `fit` still passes every other test in this file.

    Uses a per-channel CONSTANT mean field (distinguishable from the random-direction `components`,
    which are not smooth) so any confusion between "mean" and "makeup coefficient" signal would
    show up in `res.c`.
    """
    prior = _toy_prior()
    d = prior.mean.shape[0]
    third = d // 3
    mean = np.zeros(d, dtype=np.float32)
    mean[:third] = 5.0          # L
    mean[third:2 * third] = -3.0  # a
    mean[2 * third:] = 2.0        # b
    prior.mean = mean

    truth_c0 = 2.0
    d_obs = (mean + truth_c0 * prior.components[0]).astype(np.float32)

    res = fit(prior, d_obs, lam=1e-6)

    np.testing.assert_allclose(res.c[0], truth_c0, atol=1e-2)
    np.testing.assert_allclose(res.makeup, d_obs, atol=1e-2)


from makeup_transfer.priors import build_prior_from_diffs


def test_build_prior_from_diffs_recovers_a_planted_direction():
    """PCA over synthetic diffs that all lie along one direction => first component is it."""
    d = 3 * int(face_mask_flat(PRIOR_SIZE).sum())
    rng = np.random.default_rng(1)
    direction = rng.normal(size=d).astype(np.float32)
    direction /= np.linalg.norm(direction)

    amps = rng.normal(scale=3.0, size=40).astype(np.float32)
    diffs = np.stack([a * direction for a in amps])          # (40, D), rank 1

    prior = build_prior_from_diffs(diffs, var_target=0.95, max_components=8)

    assert prior.n_components >= 1
    # First component is the planted direction, up to sign.
    cos = abs(float(prior.components[0] @ direction))
    assert cos > 0.99
