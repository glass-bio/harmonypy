"""Regression for the multi-covariate boundary allocation."""

import numpy as np

from harmonypy import run_harmony


def test_covariate_bounds():
    # Boilerplate: small, repeatable input (120 cells, 5 coordinates).
    rng = np.random.default_rng(0)
    data = rng.normal(size=(120, 5)) + 1.0

    # Two correction columns: lab has 2 labels and day has 3.
    meta = {"lab": rng.permutation(np.arange(120) % 2),
            "day": rng.permutation(np.arange(120) % 3)}

    # run_harmony constructs a C++ Harmony object. It groups the labels as:
    #   [lab 0, lab 1, day 0, day 1, day 2]
    # To track which column each label belongs to, it stores running counts:
    #   B_vec = [2, 3] -> covariate_bounds = [2, 5].
    # The old constructor used resize(B_vec.size() - 1), making room for
    # only one number, then partial_sum tried to store both 2 and 5.
    # The fix, resize(B_vec.size()), makes room for both numbers.
    #
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
