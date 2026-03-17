from typing import Union

import numpy as np


ArrayLike = Union[np.ndarray, list]


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
	"""Compute classification accuracy.

	Returns the fraction of exactly correct predictions.
	"""

	y_true_arr = np.asarray(y_true).astype(int)
	y_pred_arr = np.asarray(y_pred).astype(int)

	if y_true_arr.shape != y_pred_arr.shape:
		raise ValueError("y_true and y_pred must have the same shape")

	return float((y_true_arr == y_pred_arr).mean())


def cross_entropy(y_true_one_hot: np.ndarray, y_prob: np.ndarray) -> float:
	"""Compute mean multiclass cross-entropy (log loss).

	Args:
		y_true_one_hot: Array of shape (n, C) with one-hot encoded targets.
		y_prob: Array of shape (n, C) with predicted class probabilities.
	"""

	Y = np.asarray(y_true_one_hot, dtype=np.float32)
	P = np.asarray(y_prob, dtype=np.float32)
	if Y.shape != P.shape:
		raise ValueError("y_true_one_hot and y_prob must have the same shape")
	return float(-(Y * np.log(P + 1e-12)).sum(axis=1).mean())

