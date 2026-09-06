"""Direct coordinate regressions for the existing one-covariate ridge path."""

import numpy as np
import pytest

from harmonypy import run_harmony
from .ridge_oracle import assert_coordinates, design_matrix, ridge_oracle


@pytest.mark.parametrize("counts", [(20, 20, 20), (43, 12, 5)])
@pytest.mark.parametrize("lamb", [1.0, None])
def test_one_covariate_oracle(counts, lamb):
    labels = np.repeat(np.arange(3), counts)
    data = 4 + np.random.default_rng(17).normal(size=(len(labels), 3))
    data += labels[:, None] * np.array([0.5, 1.0, 1.5])
    results = []
    for names in [np.array(["a", "b", "c"]), np.array(["z", "x", "y"])]:
        meta = {"batch": names[labels]}
        result = run_harmony(
            data, meta, ["batch"], nclust=1, theta=0,
            sigma=np.array([0.1]), lamb=lamb, batch_prop_cutoff=0,
            max_iter_harmony=1, random_state=17, ncores=1, verbose=False,
        )
        phi, _ = design_matrix(meta, ["batch"])
        np.testing.assert_array_equal(result.R, np.ones((len(labels), 1)))
        expected = ridge_oracle(data, phi, result.R, lamb=lamb)
        assert_coordinates(result.Z_corr, expected, atol=2e-5)
        results.append(result.Z_corr)
    assert_coordinates(results[0], results[1], atol=2e-5)


@pytest.mark.parametrize("lamb", [1.0, None])
def test_one_covariate_subset_oracle(lamb):
    # Three levels in distinct directions; each cluster retains exactly two.
    labels = np.repeat(np.arange(3), 20)
    data = np.array([[4., 1., 1.], [3., 2., 1.], [1., 4., 1.]])[labels]
    data += np.random.default_rng(11).normal(0, 0.1, data.shape)
    meta = {"batch": labels}
    result = run_harmony(
        data, meta, ["batch"], nclust=2, theta=0, sigma=np.array([0.3, 0.3]),
        lamb=lamb, batch_prop_cutoff=0.1, max_iter_harmony=1,
        max_iter_kmeans=0, random_state=11, ncores=1, verbose=False,
    )
    phi, _ = design_matrix(meta, ["batch"])
    fractions = (phi @ result.R) / phi.sum(axis=1)[:, None]
    keeps = [np.flatnonzero(fractions[:, k] > 0.1) for k in range(2)]
    assert any(len(keep) == 2 for keep in keeps), fractions
    keeps = [keep if len(keep) > 1 else [] for keep in keeps]
    expected = ridge_oracle(data, phi, result.R, lamb=lamb, keeps=keeps)
    assert_coordinates(result.Z_corr, expected, atol=2e-5)
