"""Regression for the multi-covariate boundary allocation."""

import numpy as np

from harmonypy import run_harmony


def test_covariate_bounds():
    # Boilerplate: small, repeatable input (120 cells, 5 coordinates).
    rng = np.random.default_rng(0)
    data = rng.normal(size=(120, 5)) + 1.0

    # The trigger is selecting TWO columns: "lab" and "day".
    # Their 2 and 3 labels give B_vec = [2, 3], so partial_sum writes [2, 5].
    # The old code allocated only one entry in covariate_bounds for these two totals.
    meta = {"lab": rng.permutation(np.arange(120) % 2),
            "day": rng.permutation(np.arange(120) % 3)}

    # The run settings keep the test short and repeatable.
    result = run_harmony(
        data, meta, ["lab", "day"], nclust=4, lamb=1.0,
        max_iter_harmony=3, max_iter_kmeans=4,
        random_state=0, ncores=1, verbose=False,
    )

    # ASan catches the original invalid write inside run_harmony, before
    # these output checks run.
    assert result.Z_corr.shape == data.shape
    assert np.isfinite(result.Z_corr).all()
