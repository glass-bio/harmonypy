"""Dense float64 reference, deliberately independent of C++ scatter kernels."""

import numpy as np


def design_matrix(meta, variables):
    blocks = []
    for var in variables:
        values = np.asarray(meta[var])
        blocks.append((values[:, None] == np.unique(values)).T.astype(float))
    return np.vstack(blocks), [len(block) for block in blocks]


def ridge_oracle(data, phi, assignments, lamb=1.0, alpha=0.2, keeps=None):
    """Correct with frozen assignments and explicit retained levels per cluster.

    A subset fits on the union of cells in retained levels, with one intercept.
    Lambda estimation uses the full cluster mass and population proportions.
    The solve uses float64, starting from the same float32 input as the backend.
    """
    z = np.asarray(data, dtype=np.float32).astype(float)
    r = np.asarray(assignments, dtype=float)
    corrected = z.copy()
    for k in range(r.shape[1]):
        keep = np.arange(len(phi)) if keeps is None else np.asarray(keeps[k])
        if len(keep) == 0:
            continue
        retained = phi[keep]
        cells = np.any(retained != 0, axis=0)
        x = np.vstack([np.ones(cells.sum()), retained[:, cells]])
        weights = r[cells, k]
        if lamb is None:
            penalties = alpha * r[:, k].sum() * phi.mean(axis=1)[keep]
        else:
            penalties = np.broadcast_to(lamb, (len(phi),))[keep]
        a = (x * weights) @ x.T + np.diag(np.r_[0.0, penalties])
        b = (x * weights) @ z[cells]
        w = np.linalg.solve(a, b)
        w[0] = 0
        corrected[cells] -= (x.T @ w) * weights[:, None]
    return corrected


def assert_coordinates(actual, expected, atol=1e-4):
    error = np.abs(actual - expected)
    relative = error / np.maximum(np.abs(expected), np.finfo(float).eps)
    print(f"max_abs={error.max():.9g} max_rel={relative.max():.9g}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)

