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
    """A prior whose components are smooth-ish random fields — stands in for real makeup."""
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
    """A difference synthesized from the basis is recovered by the fit."""
    prior = _toy_prior()
    truth = 2.0 * prior.components[1]
    res = fit(prior, truth.astype(np.float32), lam=1e-6)
    np.testing.assert_allclose(res.makeup, truth, atol=0.05)


def test_nuisance_rejection_is_the_003417_fix():
    """ZERO makeup + a global colour offset + a smooth gradient => recovered makeup ~= 0.

    This is exactly the SSAT skin-tone-drift failure. A raw chromaticity difference reports the
    whole face as makeup here; the prior fit must not.
    """
    prior = _toy_prior()
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
