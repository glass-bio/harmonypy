"""Covariate boundary safety and deterministic end-to-end execution."""

import numpy as np
import pytest

from harmonypy import run_harmony


@pytest.mark.parametrize("levels", [(2, 3), (2, 3, 5), (2, 3, 5, 4), (2, 1, 3)])
@pytest.mark.parametrize("seed", [0, 17, 42])
def test_covariate_bounds(levels, seed):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(120, 5)) + 1.0
    meta = {f"cov{i}": rng.permutation(np.arange(120) % count)
            for i, count in enumerate(levels)}
    kwargs = dict(nclust=4, lamb=1.0, max_iter_harmony=3,
                  max_iter_kmeans=4, random_state=seed, ncores=1, verbose=False)
    first = run_harmony(data, meta, list(meta), **kwargs)
    second = run_harmony(data, meta, list(meta), **kwargs)
    assert first.Z_corr.shape == data.shape
    assert np.isfinite(first.Z_corr).all()
    assert np.isfinite(first.R).all()
    np.testing.assert_array_equal(first.Z_corr, second.Z_corr)
    np.testing.assert_array_equal(first.R, second.R)
    np.testing.assert_allclose(first.R.sum(axis=1), 1, rtol=0, atol=2e-6)
