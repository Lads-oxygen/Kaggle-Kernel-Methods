from typing import Optional

import numpy as np


def _equal_channel_slices(d: int, n_channels: int) -> list[tuple[int, int]]:
	"""Return equal contiguous channel slices covering [0, d)."""

	d = int(d)
	n_channels = int(n_channels)
	if n_channels <= 0:
		raise ValueError("n_channels must be positive")
	if d % n_channels != 0:
		raise ValueError("Feature dimension must be divisible by n_channels")
	step = d // n_channels
	return [(i * step, (i + 1) * step) for i in range(n_channels)]

def augment_intercept(X: np.ndarray) -> np.ndarray:
	"""Append a constant intercept feature to each row."""

	X = np.asarray(X)
	return np.c_[X, np.ones((X.shape[0], 1), dtype=X.dtype)]


def linear_kernel_matrix(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Compute the linear kernel Gram matrix K(X, Z)."""

    X_aug = augment_intercept(np.asarray(X, dtype=np.float32))
    Z_aug = augment_intercept(np.asarray(Z, dtype=np.float32))
    d = X_aug.shape[1]
    return ((X_aug @ Z_aug.T) / d).astype(np.float32, copy=False)


def estimate_gamma(X: np.ndarray, sample: int = 500, seed: int = 0) -> float:
    """Estimate gamma for an RBF (Gaussian) kernel using the median heuristic.

    Uses gamma = 1 / (2 * median_pairwise_squared_distance).
    """

    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(sample, X.shape[0]), replace=False)
    Xs = np.asarray(X[idx], dtype=np.float32)

    G = Xs @ Xs.T
    norms = np.diag(G)
    sq = norms[:, None] + norms[None, :] - 2.0 * G
    tri = sq[np.triu_indices_from(sq, k=1)]
    med = float(np.median(tri))

    return float(1.0 / (2.0 * med + 1e-12))


def gaussian_kernel_matrix(
	X: np.ndarray,
	Z: np.ndarray,
	*,
	gamma: Optional[float] = None,
) -> np.ndarray:
	"""Compute the RBF (Gaussian) kernel Gram matrix K(X, Z).

	Args:
		gamma: If None, defaults to 1 / d where d is the augmented feature count.
	"""

	X_aug = augment_intercept(np.asarray(X, dtype=np.float32))
	Z_aug = augment_intercept(np.asarray(Z, dtype=np.float32))

	gamma_val = float(1.0 / X_aug.shape[1]) if gamma is None else float(gamma)

	Xn = np.sum(X_aug * X_aug, axis=1, keepdims=True)  # (n, 1)
	Zn = np.sum(Z_aug * Z_aug, axis=1, keepdims=True).T  # (1, m)
	sq = Xn + Zn - 2.0 * (X_aug @ Z_aug.T)

	return np.exp(-gamma_val * sq).astype(np.float32, copy=False)


def estimate_laplacian_gamma(
	X: np.ndarray, sample: int = 500, seed: int = 0, feature_block: int = 64
) -> float:
	"""Estimate gamma for a Laplacian kernel using the median heuristic.

	For k(x, z) = exp(-gamma * ||x - z||_1), a common heuristic is:
		gamma = 1 / median_pairwise_L1_distance.
	"""

	rng = np.random.default_rng(seed)
	idx = rng.choice(X.shape[0], size=min(sample, X.shape[0]), replace=False)
	Xs = np.asarray(X[idx], dtype=np.float32)

	n = Xs.shape[0]
	if n < 2:
		return 1.0

	d = Xs.shape[1]
	fs = int(feature_block)
	if fs <= 0:
		raise ValueError("feature_block must be positive")

	dist = np.zeros((n, n), dtype=np.float32)
	for j in range(0, d, fs):
		xc = Xs[:, None, j : j + fs]
		zc = Xs[None, :, j : j + fs]
		dist += np.sum(np.abs(xc - zc), axis=2)

	tri = dist[np.triu_indices_from(dist, k=1)]
	med = float(np.median(tri))
	return float(1.0 / (med + 1e-12))


def laplacian_kernel_matrix(
	X: np.ndarray,
	Z: np.ndarray,
	*,
	gamma: Optional[float] = None,
	block_size: int = 64,
	feature_block: int = 64,
) -> np.ndarray:
	"""Compute the Laplacian (L1) kernel Gram matrix K(X, Z).

	The Laplacian kernel is:
		k(x, z) = exp(-gamma * ||x - z||_1)

	Args:
		gamma: If None, defaults to 1 / d where d is the augmented feature count.
		block_size: Block size for processing rows of X.
		feature_block: Block size for processing features.
	"""

	same_inputs = Z is X
	X_aug = augment_intercept(np.asarray(X, dtype=np.float32))
	Z_aug = X_aug if same_inputs else augment_intercept(np.asarray(Z, dtype=np.float32))

	if X_aug.ndim != 2 or Z_aug.ndim != 2:
		raise ValueError("X and Z must be 2D arrays")
	if X_aug.shape[1] != Z_aug.shape[1]:
		raise ValueError("X and Z must have the same number of features")

	gamma_val = float(1.0 / X_aug.shape[1]) if gamma is None else float(gamma)

	n, d = X_aug.shape
	m = Z_aug.shape[0]
	K = np.empty((n, m), dtype=np.float32)

	bs = int(block_size)
	fs = int(feature_block)
	if bs <= 0 or fs <= 0:
		raise ValueError("block_size and feature_block must be positive")

	# Symmetry optimisation for training-time K(X, X): compute upper triangle and mirror.
	if same_inputs and n == m:
		for i0 in range(0, n, bs):
			xb = X_aug[i0 : i0 + bs]
			b = xb.shape[0]
			for j0 in range(i0, n, bs):
				zb = X_aug[j0 : j0 + bs]
				c = zb.shape[0]
				dist = np.zeros((b, c), dtype=np.float32)
				for f0 in range(0, d, fs):
					xc = xb[:, None, f0 : f0 + fs]
					zc = zb[None, :, f0 : f0 + fs]
					dist += np.sum(np.abs(xc - zc), axis=2)
				block = np.exp(-gamma_val * dist)
				K[i0 : i0 + b, j0 : j0 + c] = block
				if j0 != i0:
					K[j0 : j0 + c, i0 : i0 + b] = block.T
		return K

	# Generic (possibly non-square) cross-kernel case.
	for i in range(0, n, bs):
		xb = X_aug[i : i + bs]
		b = xb.shape[0]
		dist = np.zeros((b, m), dtype=np.float32)
		for j in range(0, d, fs):
			xc = xb[:, None, j : j + fs]
			zc = Z_aug[None, :, j : j + fs]
			dist += np.sum(np.abs(xc - zc), axis=2)
		K[i : i + b] = np.exp(-gamma_val * dist)

	return K


def estimate_chi2_gamma(
    X: np.ndarray, sample: int = 200, seed: int = 0, eps: float = 1e-12
) -> float:
    """Estimate gamma for an RBF-chi2 kernel using the median heuristic.

    Uses gamma = 1 / (2 * median_pairwise_chi2_distance).
    """

    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(sample, X.shape[0]), replace=False)
    Xs = np.asarray(X[idx], dtype=np.float32)

    # Compute chi2 distances on the sample in feature blocks.
    n = Xs.shape[0]
    d = Xs.shape[1]
    dist = np.zeros((n, n), dtype=np.float32)
    step = 64
    for j in range(0, d, step):
        xc = Xs[:, None, j : j + step]
        zc = Xs[None, :, j : j + step]
        num = (xc - zc) ** 2
        den = xc + zc + eps
        dist += 0.5 * np.sum(num / den, axis=2)

    tri = dist[np.triu_indices_from(dist, k=1)]
    med = float(np.median(tri))
    return float(1.0 / (2.0 * med + 1e-12))


def chi2_rbf_kernel_matrix(
	X: np.ndarray,
	Z: np.ndarray,
	*,
	gamma: Optional[float] = None,
	eps: float = 1e-12,
	block_size: int = 64,
	feature_block: int = 64,
) -> np.ndarray:
	"""Compute an RBF kernel based on the chi-square distance.

	The distance is:
	  d(x, z) = 0.5 * sum_i (x_i - z_i)^2 / (x_i + z_i + eps)
	and the kernel is:
	  k(x, z) = exp(-gamma * d(x, z)).

	This kernel is typically used for non-negative histogram-like features.
	"""

	same_inputs = Z is X
	X_arr = np.asarray(X, dtype=np.float32)
	Z_arr = X_arr if same_inputs else np.asarray(Z, dtype=np.float32)
	if X_arr.ndim != 2 or Z_arr.ndim != 2:
		raise ValueError("X and Z must be 2D arrays")
	if X_arr.shape[1] != Z_arr.shape[1]:
		raise ValueError("X and Z must have the same number of features")

	gamma_val = float(1.0 / X_arr.shape[1]) if gamma is None else float(gamma)

	n, d = X_arr.shape
	m = Z_arr.shape[0]
	K = np.empty((n, m), dtype=np.float32)

	bs = int(block_size)
	fs = int(feature_block)
	if bs <= 0 or fs <= 0:
		raise ValueError("block_size and feature_block must be positive")

	# Symmetry optimisation for training-time K(X, X): compute upper triangle and mirror.
	if same_inputs and n == m:
		for i0 in range(0, n, bs):
			xb = X_arr[i0 : i0 + bs]
			b = xb.shape[0]
			for j0 in range(i0, n, bs):
				zb = X_arr[j0 : j0 + bs]
				c = zb.shape[0]
				dist = np.zeros((b, c), dtype=np.float32)
				for f0 in range(0, d, fs):
					xc = xb[:, None, f0 : f0 + fs]
					zc = zb[None, :, f0 : f0 + fs]
					num = (xc - zc) ** 2
					den = xc + zc + eps
					dist += 0.5 * np.sum(num / den, axis=2)
				block = np.exp(-gamma_val * dist)
				K[i0 : i0 + b, j0 : j0 + c] = block
				if j0 != i0:
					K[j0 : j0 + c, i0 : i0 + b] = block.T
		return K

	for i in range(0, n, bs):
		xb = X_arr[i : i + bs]
		b = xb.shape[0]
		dist = np.zeros((b, m), dtype=np.float32)
		for j in range(0, d, fs):
			xc = xb[:, None, j : j + fs]
			zc = Z_arr[None, :, j : j + fs]
			num = (xc - zc) ** 2
			den = xc + zc + eps
			dist += 0.5 * np.sum(num / den, axis=2)
		K[i : i + b] = np.exp(-gamma_val * dist)

	return K


def estimate_chi2_gammas_channel(
	X: np.ndarray,
	*,
	n_channels: int = 3,
	sample: int = 200,
	seed: int = 0,
	eps: float = 1e-12,
	feature_block: int = 64,
) -> np.ndarray:
	"""Estimate per-channel gammas for `chi2_rbf_kernel_matrix_channel`.

	Uses a per-channel median heuristic:
		gamma_c = 1 / (2 * median(D_c))
	where D_c is the chi-square distance computed on channel block c.

	Assumes features are concatenated into equal-sized channel blocks.

	Args:
		n_channels: Number of channel blocks.
		sample: Number of examples to sample for the heuristic.
		seed: Random seed used for sampling.
		eps: Stability constant used in the chi-square distance.
		feature_block: Feature-block size for blockwise distance accumulation.

	Returns:
		Array of shape (n_channels,) with non-negative gamma values.
	"""

	X_arr = np.asarray(X, dtype=np.float32)
	if X_arr.ndim != 2:
		raise ValueError("X must be a 2D array")
	if feature_block <= 0:
		raise ValueError("feature_block must be positive")

	rng = np.random.default_rng(seed)
	idx = rng.choice(X_arr.shape[0], size=min(sample, X_arr.shape[0]), replace=False)
	Xs = X_arr[idx]

	n = Xs.shape[0]
	d = Xs.shape[1]
	slices = _equal_channel_slices(d, n_channels)

	gammas = np.empty((len(slices),), dtype=np.float32)
	fs = int(feature_block)

	for ch, (s, e) in enumerate(slices):
		Xc = Xs[:, s:e]
		dc_dim = Xc.shape[1]
		dist = np.zeros((n, n), dtype=np.float32)
		for j in range(0, dc_dim, fs):
			xc = Xc[:, None, j : j + fs]
			zc = Xc[None, :, j : j + fs]
			num = (xc - zc) ** 2
			den = xc + zc + eps
			dist += 0.5 * np.sum(num / den, axis=2)

		tri = dist[np.triu_indices_from(dist, k=1)]
		med = float(np.median(tri))
		gammas[ch] = float(1.0 / (2.0 * med + 1e-12))

	return gammas


def chi2_rbf_kernel_matrix_channel(
	X: np.ndarray,
	Z: np.ndarray,
	*,
	n_channels: int = 3,
	gammas: Optional[np.ndarray] = None,
	eps: float = 1e-12,
	block_size: int = 64,
	feature_block: int = 64,
) -> np.ndarray:
	"""Per-channel chi2-RBF kernel on concatenated channel blocks.

	Assumes features are concatenated as equal-sized channel blocks:
	  x = [x^(0) | x^(1) | ... | x^(C-1)]

	For each channel c, define the chi-square distance:
	  D_c(x, z) = 0.5 * sum_j (x_cj - z_cj)^2 / (x_cj + z_cj + eps)

	The kernel is:
	  K(x, z) = exp( - sum_c gamma_c * D_c(x, z) )

	Args:
		n_channels: Number of channel blocks.
		gammas: Optional non-negative gammas of shape (n_channels,). If None, defaults to gamma_c = 1 / d_c where d_c is the per-channel dimension.
		eps: Stability constant used in the chi-square distance.
		block_size: Block size for processing rows of X.
		feature_block: Block size for processing features within each channel.
  
	Returns:
		Kernel matrix of shape (n, m).
	"""

	same_inputs = Z is X
	X_arr = np.asarray(X, dtype=np.float32)
	Z_arr = X_arr if same_inputs else np.asarray(Z, dtype=np.float32)
	if X_arr.ndim != 2 or Z_arr.ndim != 2:
		raise ValueError("X and Z must be 2D arrays")
	if X_arr.shape[1] != Z_arr.shape[1]:
		raise ValueError("X and Z must have the same number of features")

	n, d = X_arr.shape
	m = Z_arr.shape[0]
	K = np.empty((n, m), dtype=np.float32)

	bs = int(block_size)
	fs = int(feature_block)
	if bs <= 0 or fs <= 0:
		raise ValueError("block_size and feature_block must be positive")

	slices = _equal_channel_slices(d, n_channels)
	dch = slices[0][1] - slices[0][0]
	if gammas is None:
		g = np.full((len(slices),), 1.0 / float(dch), dtype=np.float32)
	else:
		g = np.asarray(gammas, dtype=np.float32).reshape(-1)
		if g.shape[0] != len(slices):
			raise ValueError("gammas must have shape (n_channels,)")
		if (g < 0).any():
			raise ValueError("gammas must be non-negative")

	# Symmetry optimisation for training-time K(X, X): compute upper triangle and mirror.
	if same_inputs and n == m:
		for i0 in range(0, n, bs):
			xb_all = X_arr[i0 : i0 + bs]
			b = xb_all.shape[0]
			for j0 in range(i0, n, bs):
				zb_all = X_arr[j0 : j0 + bs]
				c = zb_all.shape[0]
				dist = np.zeros((b, c), dtype=np.float32)
				for ch, (s, e) in enumerate(slices):
					gamma_ch = float(g[ch])
					if gamma_ch == 0.0:
						continue
					xb = xb_all[:, s:e]
					zb = zb_all[:, s:e]
					dc = np.zeros((b, c), dtype=np.float32)
					dc_dim = xb.shape[1]
					for f0 in range(0, dc_dim, fs):
						xc = xb[:, None, f0 : f0 + fs]
						zc = zb[None, :, f0 : f0 + fs]
						num = (xc - zc) ** 2
						den = xc + zc + eps
						dc += 0.5 * np.sum(num / den, axis=2)
					dist += gamma_ch * dc
				block = np.exp(-dist)
				K[i0 : i0 + b, j0 : j0 + c] = block
				if j0 != i0:
					K[j0 : j0 + c, i0 : i0 + b] = block.T
		return K

	for i in range(0, n, bs):
		xb_all = X_arr[i : i + bs]
		b = xb_all.shape[0]
		dist = np.zeros((b, m), dtype=np.float32)
		for ch, (s, e) in enumerate(slices):
			if float(g[ch]) == 0.0:
				continue
			xb = xb_all[:, s:e]
			Zc = Z_arr[:, s:e]
			dc = np.zeros((b, m), dtype=np.float32)
			dc_dim = xb.shape[1]
			for j in range(0, dc_dim, fs):
				xc = xb[:, None, j : j + fs]
				zc = Zc[None, :, j : j + fs]
				num = (xc - zc) ** 2
				den = xc + zc + eps
				dc += 0.5 * np.sum(num / den, axis=2)
			dist += float(g[ch]) * dc

		K[i : i + b] = np.exp(-dist)

	return K
