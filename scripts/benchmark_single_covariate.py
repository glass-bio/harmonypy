"""Bounded synthetic compatibility/timing check; run with BLAS threads set to 1.

python scripts/benchmark_single_covariate.py --output build/before.npz
python scripts/benchmark_single_covariate.py --output build/after.npz --compare build/before.npz
"""

import argparse
import json
from time import perf_counter

import numpy as np
from harmonypy import run_harmony


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare")
    args = parser.parse_args()
    rng = np.random.default_rng(612)
    labels = rng.integers(0, 4, 3000)
    data = rng.normal(size=(len(labels), 10)) + labels[:, None] * 0.2

    def run():
        return run_harmony(
            data, {"batch": labels}, ["batch"], nclust=12, theta=2,
            lamb=1.0, max_iter_harmony=3, max_iter_kmeans=4,
            random_state=612, ncores=1, verbose=False,
        )

    run()  # Warm up BLAS and allocation.
    times = []
    for _ in range(7):
        start = perf_counter()
        result = run()
        times.append(perf_counter() - start)
    z, r = result.Z_corr, result.R
    assert np.isfinite(z).all() and np.isfinite(r).all()
    np.savez(args.output, coordinates=z, assignments=r, seconds=times)
    report = {"seconds": times, "median_seconds": float(np.median(times)),
              "shape": list(z.shape), "coordinate_sum": float(z.sum())}
    if args.compare:
        before = np.load(args.compare)
        report["max_coordinate_change"] = float(np.abs(z - before["coordinates"]).max())
        report["max_assignment_change"] = float(np.abs(r - before["assignments"]).max())
        report["median_ratio"] = float(np.median(times) / np.median(before["seconds"]))
        np.testing.assert_allclose(z, before["coordinates"], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(r, before["assignments"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
