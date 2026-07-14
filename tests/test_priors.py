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
