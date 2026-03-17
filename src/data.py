from pathlib import Path
from typing import Dict, Hashable, Tuple

import numpy as np
import pandas as pd


def load_data(
	data_dir: str | Path,
	*,
	n_features: int = 3072,
	xtr_name: str = "Xtr.csv",
	xte_name: str = "Xte.csv",
	ytr_name: str = "Ytr.csv",
	y_col: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Load the challenge data from CSV files.

	Expected files:
	- Xtr.csv: training features (n_train, n_features)
	- Xte.csv: test features (n_test, n_features)
	- Ytr.csv: training labels (n_train, 2) where labels are in column `y_col`

	Returns:
		X_tr, X_te, y_tr_raw
	"""

	data_path = Path(data_dir)

	X_tr = pd.read_csv(
		data_path / xtr_name,
		header=None,
		sep=",",
		usecols=range(n_features),
	).to_numpy()

	X_te = pd.read_csv(
		data_path / xte_name,
		header=None,
		sep=",",
		usecols=range(n_features),
	).to_numpy()

	y_tr = (
		pd.read_csv(data_path / ytr_name, sep=",", usecols=[y_col])
		.to_numpy()
		.squeeze()
	)

	return np.asarray(X_tr), np.asarray(X_te), np.asarray(y_tr)


def train_val_split(
	X: np.ndarray,
	y: np.ndarray,
	*,
	test_size: float = 0.2,
	seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Simple shuffled train/validation split.

	Args:
		X: Feature matrix of shape (n, d)
		y: Labels of shape (n,)
		test_size: Fraction of samples to use for validation
		seed: RNG seed
	"""

	if not (0.0 < test_size < 1.0):
		raise ValueError("test_size must be in (0, 1)")

	n = int(X.shape[0])
	if y.shape[0] != n:
		raise ValueError("X and y must have the same number of rows")

	idx = np.arange(n)
	rng = np.random.default_rng(seed)
	rng.shuffle(idx)

	cut = int(n * (1.0 - test_size))
	tr_idx, va_idx = idx[:cut], idx[cut:]

	return X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, Hashable]]:
	"""Encode arbitrary labels into integer class ids.

	Returns the integer-encoded labels and a map
	from integer id -> original label (useful for submissions).
	"""

	classes = np.unique(y)
	to_int = {c: i for i, c in enumerate(classes)}
	to_lbl: Dict[int, Hashable] = {i: c for c, i in to_int.items()}
	y_int = np.array([to_int[v] for v in y], dtype=int)
	return y_int, to_lbl

