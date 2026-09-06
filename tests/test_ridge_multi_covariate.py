"""Conditional weighted ridge, with clustering and pruning held fixed."""

from itertools import permutations, product

import numpy as np
import pytest

from harmonypy import run_harmony
from .ridge_oracle import assert_coordinates, design_matrix, ridge_oracle


def run_once(data, meta, variables, lamb, **kwargs):
    options = dict(nclust=1, theta=0, sigma=np.array([0.1]), lamb=lamb,
                   batch_prop_cutoff=0, max_iter_harmony=1,
                   random_state=27, ncores=1, verbose=False)
    options.update(kwargs)
    return run_harmony(data, meta, variables, **options)


def variants(meta):
    # All covariate orders and a bijective reordering of each level encoding.
    relabeled = {}
    for name, values in meta.items():
        levels, codes = np.unique(values, return_inverse=True)
        relabeled[name] = np.array([f"label{len(levels)-i}" for i in range(len(levels))])[codes]
    for variables in permutations(meta):
        for encoded in [meta, relabeled]:
            yield encoded, list(variables)


@pytest.mark.parametrize("lamb", [1.0, None])
def test_binary_pair_reproducer(lamb):
    pairs = np.repeat(np.array(list(product(range(2), repeat=2))), [30, 5, 5, 10], axis=0)
    data = (10 + 2 * pairs[:, 0] + 3 * pairs[:, 1])[:, None].astype(float)
    meta = dict(a=pairs[:, 0], b=pairs[:, 1])
    baseline = None
    for encoded, variables in variants(meta):
        result = run_once(data, encoded, variables, lamb)
        phi, _ = design_matrix(encoded, variables)
        expected = ridge_oracle(data, phi, result.R, lamb=lamb)
        print(f"binary lambda={lamb} corrected_groups={result.Z_corr[[0,30,35,40],0].tolist()}")
        assert_coordinates(result.Z_corr, expected)
        if baseline is not None:
            assert_coordinates(result.Z_corr, baseline)
        if baseline is None:
            baseline = result.Z_corr


def scenario(kind, n_covariates):
    grid = np.array(list(product(range(2), range(3), range(4))))
    if kind == "balanced":
        labels = np.repeat(grid, 2, axis=0)
    elif kind == "unbalanced":
        labels = np.repeat(grid, 1 + 2 * (grid[:, 0] == 0) + (grid[:, 2] == 0), axis=0)
    elif kind == "nested":
        fine = np.tile(np.arange(4), 12)
        labels = np.column_stack([fine // 2, fine, fine % 2])
    else:
        fine = np.tile(np.arange(3), 16)
        labels = np.column_stack([fine, fine, fine])
    return {f"cov{i}": labels[:, i] for i in range(n_covariates)}


@pytest.mark.parametrize("kind", ["balanced", "unbalanced", "nested", "duplicate"])
@pytest.mark.parametrize("n_covariates", [1, 2, 3])
@pytest.mark.parametrize("n_pcs", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("lamb", [1.0, None])
def test_covariate_matrix(kind, n_covariates, n_pcs, lamb):
    meta = scenario(kind, n_covariates)
    labels = np.column_stack(list(meta.values()))
    rng = np.random.default_rng(201)
    data = 3 + rng.normal(0, 0.4, (len(labels), n_pcs))
    data += labels @ rng.uniform(0.1, 0.6, (n_covariates, n_pcs))
    baseline = None
    for encoded, variables in variants(meta):
        result = run_once(data, encoded, variables, lamb)
        phi, _ = design_matrix(encoded, variables)
        expected = ridge_oracle(data, phi, result.R, lamb=lamb)
        assert_coordinates(result.Z_corr, expected)
        if baseline is not None:
            assert_coordinates(result.Z_corr, baseline)
        if baseline is None:
            baseline = result.Z_corr


@pytest.mark.parametrize("n_covariates", [2, 3])
@pytest.mark.parametrize("lamb", [1.0, None])
def test_soft_cluster_weights(n_covariates, lamb):
    meta = scenario("unbalanced", n_covariates)
    data = np.random.default_rng(73).normal(size=(len(next(iter(meta.values()))), 5))
    baseline = None
    for encoded, variables in variants(meta):
        result = run_once(data, encoded, variables, lamb, nclust=3, sigma=np.full(3, 0.7))
        assert np.any((result.R > 0.2) & (result.R < 0.8))
        phi, _ = design_matrix(encoded, variables)
        expected = ridge_oracle(data, phi, result.R, lamb=lamb)
        assert_coordinates(result.Z_corr, expected)
        if baseline is not None:
            assert_coordinates(result.Z_corr, baseline)
        if baseline is None:
            baseline = result.Z_corr


@pytest.mark.parametrize("n_covariates", [1, 2, 3])
@pytest.mark.parametrize("n_pcs", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("lamb", [1.0, None])
def test_pruned_subsets(n_covariates, n_pcs, lamb):
    # Each of two well-separated groups has two levels in the first covariate.
    # Covariate 2 is entirely removed for group 1 (only one qualifying level).
    # Covariate 3 duplicates covariate 1. These selections are unchanged by
    # either the current denominator or the true per-level population count.
    a = np.repeat(np.arange(4), 12)
    group = a // 2
    meta = dict(a=a)
    if n_covariates >= 2:
        meta["b"] = np.where(group == 0, a, 2)
    if n_covariates == 3:
        meta["c"] = a.copy()
    centers = np.ones((2, n_pcs))
    centers[:, 0] = [5, -5]
    data = centers[group] + np.random.default_rng(83).normal(0, 0.15, (48, n_pcs))
    baseline = None
    for encoded, variables in variants(meta):
        result = run_once(data, encoded, variables, lamb, nclust=2,
                          sigma=np.full(2, 0.1), batch_prop_cutoff=0.3,
                          max_iter_kmeans=0)
        phi, sizes = design_matrix(encoded, variables)
        keeps = []
        for k in range(2):
            cluster_group = int(result.R[group == 1, k].mean() > 0.5)
            assert result.R[group == cluster_group, k].mean() > 0.99
            # Expected membership follows the known fixture groups, not C++.
            eligible = (phi @ (group == cluster_group)) / phi.sum(axis=1) > 0.9
            offset = 0
            for size in sizes:
                if eligible[offset:offset + size].sum() < 2:
                    eligible[offset:offset + size] = False
                offset += size
            keep = np.flatnonzero(eligible)
            assert 0 < len(keep) < len(phi)
            keeps.append(keep)
        expected = ridge_oracle(data, phi, result.R, lamb=lamb, keeps=keeps)
        assert_coordinates(result.Z_corr, expected)
        if baseline is not None:
            assert_coordinates(result.Z_corr, baseline)
        if baseline is None:
            baseline = result.Z_corr


@pytest.mark.parametrize("n_covariates", [2, 3])
@pytest.mark.parametrize("lamb", [1.0, None])
def test_partially_overlapping_pruned_levels(n_covariates, lamb):
    # Retained rows overlap only partially: cells may have zero, one, or
    # several retained levels. This distinguishes a union from double counting
    # and from fitting the intercept on every cell regardless of pruning.
    a = np.r_[np.repeat(np.arange(4), [36, 36, 4, 4]),
              np.repeat(np.arange(4), [4, 4, 36, 36])]
    group = np.repeat([0, 1], 80)
    rng = np.random.default_rng(92)
    b = np.r_[rng.permutation(a[:80]), rng.permutation(a[80:])]
    meta = dict(a=a, b=b)
    if n_covariates == 3:
        meta["c"] = a.copy()
    data = np.column_stack([np.where(group == 0, 5., -5.), np.ones(160)])
    data += rng.normal(0, 0.1, data.shape)
    baseline = None
    for encoded, variables in variants(meta):
        result = run_once(data, encoded, variables, lamb, nclust=2,
                          sigma=np.full(2, 0.4), batch_prop_cutoff=0.75,
                          max_iter_kmeans=0)
        phi, _ = design_matrix(encoded, variables)
        keeps = []
        for k in range(2):
            cluster_group = int(result.R[group == 1, k].mean() > 0.5)
            keep = np.flatnonzero((phi @ (group == cluster_group)) / phi.sum(axis=1) > 0.75)
            memberships = phi[keep].sum(axis=0)
            assert set(memberships) >= {0, 1, n_covariates}
            keeps.append(keep)
        expected = ridge_oracle(data, phi, result.R, lamb=lamb, keeps=keeps)
        assert_coordinates(result.Z_corr, expected)
        if baseline is not None:
            assert_coordinates(result.Z_corr, baseline)
        if baseline is None:
            baseline = result.Z_corr


@pytest.mark.parametrize("kind", ["balanced", "unbalanced", "nested", "duplicate"])
@pytest.mark.parametrize("n_covariates", [1, 2, 3])
@pytest.mark.parametrize("lamb", [1.0, None])
def test_all_level_permutations(kind, n_covariates, lamb):
    meta = scenario(kind, n_covariates)
    data = 3 + np.random.default_rng(401).normal(0, 0.3, (len(next(iter(meta.values()))), 3))
    phi, sizes = design_matrix(meta, list(meta))
    expected = ridge_oracle(data, phi, np.ones((len(data), 1)), lamb=lamb)
    reference = run_once(data, meta, list(meta), lamb).Z_corr
    max_oracle = max_permutation = 0.0
    # Exhaust every bijection of labels jointly, and every covariate order.
    # Small synthetic matrices keep this exhaustive regression bounded.
    for level_orders in product(*(permutations(range(size)) for size in sizes)):
        encoded = {name: np.asarray(order)[np.unique(values, return_inverse=True)[1]]
                   for (name, values), order in zip(meta.items(), level_orders)}
        for variables in permutations(meta):
            actual = run_once(data, encoded, list(variables), lamb).Z_corr
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-4)
            max_oracle = max(max_oracle, float(np.abs(actual - expected).max()))
            max_permutation = max(max_permutation, float(np.abs(actual - reference).max()))
    print(f"exhaustive max_oracle={max_oracle:.9g} max_permutation={max_permutation:.9g}")


@pytest.mark.parametrize("n_covariates,prune", [
    (2, False), (2, True), (3, True),
    pytest.param(3, False, marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="Existing float32 inverse exceeds 1e-4 for this fixture (max 1.15932397e-4)",
    )),
])
def test_level_specific_penalties(n_covariates, prune):
    a = np.r_[np.repeat(np.arange(4), [36, 36, 4, 4]),
              np.repeat(np.arange(4), [4, 4, 36, 36])]
    group = np.repeat([0, 1], 80)
    rng = np.random.default_rng(93)
    meta = dict(a=a, b=np.r_[rng.permutation(a[:80]), rng.permutation(a[80:])])
    if n_covariates == 3:
        # With pruning this last covariate has only one qualifying level and
        # is removed entirely; other cases remove middle covariates instead.
        meta["c"] = a // 2
    data = np.column_stack([np.where(group == 0, 5., -5.), np.ones(160), np.ones(160)])
    data += rng.normal(0, 0.1, data.shape)
    baseline = None
    for encoded, variables in variants(meta):
        penalties = []
        for name in variables:
            for level in np.unique(encoded[name]):
                original_label = np.unique(meta[name][encoded[name] == level]).item()
                penalties.append(0.75 + 0.25 * (list(meta).index(name) + original_label))
        result = run_once(data, encoded, variables, penalties, nclust=2,
                          sigma=np.full(2, 0.4), batch_prop_cutoff=0.75 if prune else 0,
                          max_iter_kmeans=0)
        phi, sizes = design_matrix(encoded, variables)
        keeps = None
        if prune:
            keeps = []
            for k in range(2):
                cluster_group = int(result.R[group == 1, k].mean() > 0.5)
                eligible = (phi @ (group == cluster_group)) / phi.sum(axis=1) > 0.75
                offset = 0
                for size in sizes:
                    if eligible[offset:offset + size].sum() < 2:
                        eligible[offset:offset + size] = False
                    offset += size
                keep = np.flatnonzero(eligible)
                assert len(keep) == 4
                keeps.append(keep)
        assert np.isfinite(result.R).all() and np.isfinite(result.Z_corr).all()
        expected = ridge_oracle(data, phi, result.R, lamb=penalties, keeps=keeps)
        assert_coordinates(result.Z_corr, expected)
        if baseline is None:
            baseline = result.Z_corr
        else:
            assert_coordinates(result.Z_corr, baseline)
