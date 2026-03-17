import numpy as np


class StandardiseData:
    """Standardise features by removing mean and scaling to unit variance.

	`sigma` is regularised by +1e-12.
	"""

    def __init__(self):
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardiseData":
        X = np.asarray(X)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-12
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise ValueError("StandardiseData not fitted (call fit first).")
        X = np.asarray(X)
        return (X - self.mu) / self.sigma


CentreData = StandardiseData


class PCA:
    """Principal component analysis (PCA) projection, with optional whitening.

    Fit learns a mean vector and an orthonormal basis (principal components)
    using an SVD of the centred training data.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        whiten: bool = False,
        eps: float = 1e-12,
    ):
        """Create a PCA transformer.

        Args:
            n_components: Number of components to keep. If None, keeps all
                possible components (up to min(n_samples, n_features)).
            whiten: If True, scales projected components to unit variance.
            eps: Numerical stabiliser used when whitening.
        """

        self.n_components = n_components
        self.whiten = bool(whiten)
        self.eps = float(eps)

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # (k, d)
        self.singular_values_: np.ndarray | None = None  # (k,)
        self.n_samples_: int | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Learn PCA parameters from training data.

        Args:
            X: Training features of shape (n, d).

        Returns:
            Self.
        """

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")

        n, d = X_arr.shape
        self.n_samples_ = int(n)

        mean = X_arr.mean(axis=0)
        Xc = X_arr - mean

        # SVD: Xc = U S Vt
        # Principal directions are rows of Vt.
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        max_components = int(min(n, d))
        k = max_components if self.n_components is None else int(self.n_components)
        if not (1 <= k <= max_components):
            raise ValueError(f"n_components must be in [1, {max_components}]")

        self.mean_ = mean.astype(np.float32, copy=False)
        self.components_ = Vt[:k].astype(np.float32, copy=False)
        self.singular_values_ = S[:k].astype(np.float32, copy=False)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into the learned PCA space.

        Args:
            X: Features of shape (m, d).

        Returns:
            Projected features of shape (m, k).
        """

        if (
            self.mean_ is None
            or self.components_ is None
            or self.singular_values_ is None
            or self.n_samples_ is None
        ):
            raise ValueError("PCA not fitted (call fit first).")

        X_arr = np.asarray(X, dtype=np.float32)
        Xc = X_arr - self.mean_
        Z = Xc @ self.components_.T

        if self.whiten:
            scale = np.sqrt(max(self.n_samples_ - 1, 1)) / (self.singular_values_ + self.eps)
            Z = Z * scale[None, :]

        return Z
