"""Direct-coordinate checks for ridge conditioning, with frozen assignments."""

from itertools import permutations, product

import numpy as np
import pytest

from harmonypy import run_harmony
from .ridge_oracle import design_matrix, ridge_oracle


def run_once(data, meta, variables, penalties, **kwargs):
    options = dict(nclust=1, theta=0, sigma=np.array([0.1]), lamb=penalties,
                   batch_prop_cutoff=0, max_iter_harmony=1, max_iter_kmeans=0,
                   random_state=27, ncores=1, verbose=False)
    options.update(kwargs)
    return run_harmony(data, meta, variables, **options)


def encodings(meta):
    renamed = {}
    for name, values in meta.items():
        levels, codes = np.unique(values, return_inverse=True)
        renamed[name] = np.asarray([f"level{len(levels)-i}" for i in range(len(levels))])[codes]
    for variables in permutations(meta):
        for labels in (meta, renamed):
            yield labels, list(variables)


def check_oracle(data, meta, variables, result, penalties, atol=1e-4):
    phi, _ = design_matrix(meta, variables)
    expected = ridge_oracle(data, phi, result.R, lamb=penalties)
    np.testing.assert_allclose(result.Z_corr, expected, rtol=0, atol=atol)
    return result.Z_corr


@pytest.mark.parametrize("weak", [False, True])
def test_binary_pair_coordinates(weak):
    counts = (3000, 500, 500, 1000) if weak else (30, 5, 5, 10)
    pairs = np.repeat(np.asarray(list(product(range(2), repeat=2))), counts, axis=0)
    data = (10 + 2 * pairs[:, 0] + 3 * pairs[:, 1])[:, None].astype(float)
    meta = dict(a=pairs[:, 0], b=pairs[:, 1])
    penalty = 0.01 if weak else 1.0
    reference = None
    for labels, variables in encodings(meta):
        result = run_once(data, labels, variables, penalty)
        assert np.array_equal(result.R, np.ones((len(data), 1)))
        actual = check_oracle(data, labels, variables, result, penalty)
        if reference is not None:
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-4)
        reference = actual



@pytest.mark.parametrize("weak", [False, True])
def test_one_covariate_coordinates(weak):
    counts = (3000, 500, 500, 1000) if weak else (30, 5, 5, 10)
    pairs = np.repeat(np.asarray(list(product(range(2), repeat=2))), counts, axis=0)
    data = (10 + 2 * pairs[:, 0] + 3 * pairs[:, 1])[:, None].astype(float)
    meta = dict(a=pairs[:, 0])
    penalty = 0.01 if weak else 1.0
    reference = None
    for labels in (meta, dict(a=np.where(meta["a"] == 0, "z", "a"))):
        result = run_once(data, labels, ["a"], penalty)
        actual = check_oracle(data, labels, ["a"], result, penalty)
        if reference is not None:
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-4)
        reference = actual


def test_auto_penalty_coordinates():
    pairs = np.repeat(np.asarray(list(product(range(2), repeat=2))),
                      (30, 5, 5, 10), axis=0)
    data = (10 + 2 * pairs[:, 0] + 3 * pairs[:, 1])[:, None].astype(float)
    meta = dict(a=pairs[:, 0], b=pairs[:, 1])
    for labels, variables in encodings(meta):
        result = run_once(data, labels, variables, None)
        check_oracle(data, labels, variables, result, None)
    result = run_once(data, dict(a=meta["a"]), ["a"], None)
    check_oracle(data, dict(a=meta["a"]), ["a"], result, None)


def test_level_specific_penalties():
    a = np.r_[np.repeat(np.arange(4), [36, 36, 4, 4]),
              np.repeat(np.arange(4), [4, 4, 36, 36])]
    group = np.repeat([0, 1], 80)
    rng = np.random.default_rng(93)
    meta = dict(a=a, b=np.r_[rng.permutation(a[:80]), rng.permutation(a[80:])],
                c=a // 2)
    data = np.column_stack([np.where(group == 0, 5., -5.), np.ones(160), np.ones(160)])
    data += rng.normal(0, 0.1, data.shape)
    reference = None
    for labels, variables in encodings(meta):
        penalties = []
        for name in variables:
            for level in np.unique(labels[name]):
                original = np.unique(meta[name][labels[name] == level]).item()
                penalties.append(0.75 + 0.25 * (list(meta).index(name) + original))
        result = run_once(data, labels, variables, penalties, nclust=2,
                          sigma=np.full(2, 0.4))
        actual = check_oracle(data, labels, variables, result, penalties)
        if reference is not None:
            np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-4)
        reference = actual


def test_unsupported_weak_penalty_fails_loudly():
    pairs = np.repeat(np.asarray(list(product(range(2), repeat=2))),
                      (3000, 500, 500, 1000), axis=0)
    data = (10 + 2 * pairs[:, 0] + 3 * pairs[:, 1])[:, None].astype(float)
    with pytest.raises(RuntimeError, match="ridge accuracy"):
        run_once(data, dict(a=pairs[:, 0], b=pairs[:, 1]), ["a", "b"], 1e-12)
