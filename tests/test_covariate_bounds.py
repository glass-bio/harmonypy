"""Regression for the multi-covariate boundary allocation."""

import numpy as np

from harmonypy import run_harmony


def test_covariate_bounds():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(120, 5)) + 1.0
    meta = {"lab": rng.permutation(np.arange(120) % 2),
            "day": rng.permutation(np.arange(120) % 3)}
    result = run_harmony(
        data, meta, ["lab", "day"], nclust=4, lamb=1.0,
        max_iter_harmony=3, max_iter_kmeans=4,
        random_state=0, ncores=1, verbose=False,
    )
    assert result.Z_corr.shape == data.shape
    assert np.isfinite(result.Z_corr).all()
