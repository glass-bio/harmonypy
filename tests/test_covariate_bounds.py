"""Harmony should accept multiple correction columns for the same cells."""

import numpy as np

from harmonypy import run_harmony


def test_run_harmony_with_lab_and_day():
    """Correcting for lab and day should return finite coordinates for every cell."""
    n_cells = 120
    rng = np.random.default_rng(0)
    cell_coordinates = rng.normal(size=(n_cells, 5)) + 1.0

    # Model cells processed in two labs across three days. Each cell has
    # both a lab label and a day label, which we want to correct for together.
    metadata = {
        "lab": rng.permutation(np.tile(["lab_a", "lab_b"], n_cells // 2)),
        "day": rng.permutation(np.tile(["Monday", "Tuesday", "Wednesday"], n_cells // 3)),
    }

    # Keep both columns: one column takes a different constructor path and
    # does not exercise storing the running label counts for multiple columns.
    # The remaining settings keep the run short and repeatable.
    result = run_harmony(
        cell_coordinates, metadata, ["lab", "day"], nclust=4, lamb=1.0,
        max_iter_harmony=3, max_iter_kmeans=4,
        random_state=0, ncores=1, verbose=False,
    )

    # Every input cell should still have five coordinates, all finite.
    assert result.Z_corr.shape == cell_coordinates.shape
    assert np.isfinite(result.Z_corr).all()
