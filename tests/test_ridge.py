"""Check ridge correction against a dense float64 least-squares solution."""

from itertools import permutations, product

import numpy as np
import pytest

from harmonypy import run_harmony


def correct_once(data, meta, variables, lamb, **kwargs):
    # One correction step lets the reference use the same cluster assignments.
    options = dict(
        nclust=1, theta=0, sigma=np.array([0.1]), lamb=lamb,
        batch_prop_cutoff=0, max_iter_harmony=1,
        random_state=27, ncores=1, verbose=False,
    )
    options.update(kwargs)
    return run_harmony(data, meta, variables, **options)


def design_matrix(meta, variables):
    blocks = [(meta[name][:, None] == np.unique(meta[name])).T
              for name in variables]
    return np.vstack(blocks).astype(float), [len(block) for block in blocks]


def ridge_reference(data, phi, assignments, lamb, keeps=None):
    # Start from the backend's float32 input, then solve independently in float64.
    z = np.asarray(data, dtype=np.float32).astype(float)
    corrected = z.copy()
    for k, weights in enumerate(assignments.T):
        keep = np.arange(len(phi)) if keeps is None else keeps[k]
        if len(keep) == 0:
            continue
        retained = phi[keep]
        cells = np.any(retained != 0, axis=0)
        x = np.vstack([np.ones(cells.sum()), retained[:, cells]])
        if lamb is None:
            penalties = 0.2 * weights.sum() * phi.mean(axis=1)[keep]
        else:
            penalties = np.broadcast_to(lamb, (len(phi),))[keep]
        weighted_x = x * weights[cells]
        a = weighted_x @ x.T + np.diag(np.r_[0.0, penalties])
        b = weighted_x @ z[cells]
        coefficients = np.linalg.solve(a, b)
        coefficients[0] = 0  # Keep the intercept in the corrected coordinates.
        corrected[cells] -= (x.T @ coefficients) * weights[cells, None]
    return corrected


def encodings(meta):
    # Relabeling levels and reordering columns should preserve the correction.
    relabeled = {}
    for name, values in meta.items():
        levels, codes = np.unique(values, return_inverse=True)
        relabeled[name] = (len(levels) - 1 - codes)
    for variables in permutations(meta):
        for encoded in (meta, relabeled):
            yield encoded, list(variables)


@pytest.mark.parametrize("lamb", [1.0, None])
def test_ridge_with_lab_and_day(lamb):
    # Unequal group sizes expose both missing lab/day intersections and an
    # intercept counted once per column instead of once per cell.
    pairs = np.repeat(list(product(range(2), repeat=2)), [30, 5, 5, 10], axis=0)
    data = (10 + 2 * pairs[:, 0] + 3 * pairs[:, 1])[:, None].astype(float)
    meta = dict(lab=pairs[:, 0], day=pairs[:, 1])
    for encoded, variables in encodings(meta):
        result = correct_once(data, encoded, variables, lamb)
        np.testing.assert_array_equal(result.R, np.ones((50, 1)))
        phi, _ = design_matrix(encoded, variables)
        expected = ridge_reference(data, phi, result.R, lamb)
        if lamb == 1.0:
            np.testing.assert_allclose(
                expected[[0, 30, 35, 40], 0],
                [12.42424242, 12.54545455, 12.45454545, 12.57575758],
                rtol=0, atol=1e-8,
            )
        np.testing.assert_allclose(result.Z_corr, expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("kind", ["unbalanced", "nested", "duplicate"])
@pytest.mark.parametrize("n_covariates", [1, 2, 3])
@pytest.mark.parametrize("n_clusters", [1, 3])
@pytest.mark.parametrize("penalty", ["fixed", "automatic", "per_level"])
def test_ridge_covariates(kind, n_covariates, n_clusters, penalty):
    if kind == "unbalanced":
        grid = np.array(list(product(range(2), range(3), range(4))))
        labels = np.repeat(grid, 1 + 2 * (grid[:, 0] == 0), axis=0)
    elif kind == "nested":
        fine = np.tile(np.arange(4), 12)
        labels = np.column_stack([fine // 2, fine, fine % 2])
    else:
        fine = np.tile(np.arange(3), 16)
        labels = np.column_stack([fine, fine, fine])
    meta = {f"cov{i}": labels[:, i] for i in range(n_covariates)}
    data = 3 + np.random.default_rng(201).normal(0, 0.4, (len(labels), 3))
    baseline = None
    for encoded, variables in encodings(meta):
        lamb = None if penalty == "automatic" else 1.0
        if penalty == "per_level":
            # Follow each original level when its encoding/order changes.
            lamb = [1 + 0.25 * np.unique(meta[name][encoded[name] == level]).item()
                    for name in variables for level in np.unique(encoded[name])]
        result = correct_once(
            data, encoded, variables, lamb,
            nclust=n_clusters, sigma=np.full(n_clusters, 0.7),
        )
        if n_clusters > 1:
            assert np.any((result.R > 0.2) & (result.R < 0.8))
        phi, _ = design_matrix(encoded, variables)
        expected = ridge_reference(data, phi, result.R, lamb)
        np.testing.assert_allclose(result.Z_corr, expected, rtol=0, atol=1e-4)
        if baseline is None:
            baseline = result.Z_corr
        else:
            np.testing.assert_allclose(result.Z_corr, baseline, rtol=0, atol=1e-4)


@pytest.mark.parametrize("n_covariates", [2, 3])
@pytest.mark.parametrize("lamb", [1.0, None])
def test_ridge_pruned_levels(n_covariates, lamb):
    a = np.r_[np.repeat(np.arange(4), [36, 36, 4, 4]),
              np.repeat(np.arange(4), [4, 4, 36, 36])]
    group = np.repeat([0, 1], 80)
    rng = np.random.default_rng(92)
    meta = dict(a=a, b=np.r_[rng.permutation(a[:80]), rng.permutation(a[80:])])
    if n_covariates == 3:
        # Only one level qualifies in each cluster, so this column is removed.
        meta["c"] = group.copy()
    data = np.column_stack([np.where(group == 0, 5., -5.), np.ones(160)])
    data += rng.normal(0, 0.1, data.shape)
    for encoded, variables in encodings(meta):
        result = correct_once(
            data, encoded, variables, lamb, nclust=2, sigma=np.full(2, 0.4),
            batch_prop_cutoff=0.75, max_iter_kmeans=0,
        )
        phi, sizes = design_matrix(encoded, variables)
        keeps = []
        for k in range(2):
            cluster_group = int(result.R[group == 1, k].mean() > 0.5)
            assert result.R[group == cluster_group, k].mean() > 0.99
            # Expected retained levels follow the fixture's known groups.
            # The selection is the same with the current pruning denominator
            # and with actual per-level cell counts.
            eligible = (phi @ (group == cluster_group)) / phi.sum(axis=1) > 0.75
            offset = 0
            for size in sizes:
                if eligible[offset:offset + size].sum() < 2:
                    eligible[offset:offset + size] = False
                offset += size
            keep = np.flatnonzero(eligible)
            # The intercept must count a union: zero, one or two retained
            # levels can contain a cell, even after a column is removed.
            assert len(keep) == 4
            assert set(phi[keep].sum(axis=0)) == {0, 1, 2}
            keeps.append(keep)
        expected = ridge_reference(data, phi, result.R, lamb, keeps)
        np.testing.assert_allclose(result.Z_corr, expected, rtol=0, atol=1e-4)


def test_ridge_without_retained_covariates():
    # A column with only one level cannot contribute a batch effect.
    data = np.random.default_rng(19).normal(size=(40, 3))
    meta = dict(lab=np.zeros(40), day=np.ones(40))
    result = correct_once(data, meta, list(meta), lamb=1.0)
    np.testing.assert_array_equal(result.Z_corr, data.astype(np.float32))
