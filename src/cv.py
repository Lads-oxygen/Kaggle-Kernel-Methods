from __future__ import annotations

from typing import List

import numpy as np


def make_folds(n: int, k: int = 5, seed: int = 0) -> List[np.ndarray]:
    """Create k random folds of indices.

    Args:
        n: Number of samples.
        k: Number of folds.
        seed: RNG seed.

    Returns:
        List of length k; each element is an int array of indices.
    """

    if k <= 1:
        raise ValueError("k must be >= 2")
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.RandomState(int(seed))
    idx = rng.permutation(int(n))
    folds = np.array_split(idx, int(k))
    return [f.astype(int, copy=False) for f in folds]


def fold_train_val_indices(folds: List[np.ndarray], fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) for a given fold index."""

    k = len(folds)
    if not (0 <= fold < k):
        raise ValueError("fold must be in [0, k)")

    val_idx = folds[fold]
    tr_idx = np.concatenate([folds[i] for i in range(k) if i != fold]).astype(int, copy=False)
    return tr_idx, val_idx
