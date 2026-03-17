from typing import Callable, Optional, Tuple

import numpy as np


KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class KernelSVM:
    """Multiclass kernel SVM trained in the dual.

    Uses a one-vs-rest (OvR) construction with a shared training Gram matrix.
    Each class k learns signed dual coefficients a_k such that the score is:
        s_k(x) = sum_i a_{ik} K(x_i, x)

    The optimisation uses the (bias-free) dual formulation. For binary labels y_i in {-1, +1}, the objective is:
        maximise  2 * sum_i a_i y_i  -  a^T K a
        subject to  0 <= y_i a_i <= 1 / (2 * lam * n)

    We optimise with projected gradient ascent on the concave quadratic.
    """

    def __init__(
        self,
        n_classes: int,
        *,
        kernel_fn: Optional[KernelFn] = None,
        lr: float = 1.0,
        epochs: int = 100,
        lam: float = 1e-4,
        patience: int = 10,
    ):
        """Create a kernel SVM estimator.

        Args:
            n_classes: Number of classes.
            kernel_fn: Function returning the Gram matrix K(X, Z). If None, you
                must pass precomputed kernels via fit(..., K=...) and
                predict(..., K_star=...).
            lr: Step size multiplier (internally scaled by a safe bound).
            epochs: Maximum number of optimisation steps.
            lam: Regularisation strength (appears in the dual box constraint).
            patience: Early stopping patience (in epochs) when validation is used.
        """

        self.n_classes = int(n_classes)
        self.kernel_fn = kernel_fn
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.lam = float(lam)
        self.patience = int(patience)

        self.X_train_: Optional[np.ndarray] = None
        self.A_: Optional[np.ndarray] = None  # (n, C) signed dual coefficients

        self._K_col_mean_: Optional[np.ndarray] = None
        self._K_mean_: Optional[float] = None

    def _centre_train_gram(self, K: np.ndarray) -> np.ndarray:
        row_mean = K.mean(axis=1, keepdims=True)  # (n, 1)
        col_mean = K.mean(axis=0, keepdims=True)  # (1, n)
        mean = float(K.mean())

        self._K_col_mean_ = col_mean
        self._K_mean_ = mean

        return K - row_mean - col_mean + mean

    def _centre_cross_gram(self, Kxz: np.ndarray) -> np.ndarray:
        if self._K_col_mean_ is None or self._K_mean_ is None:
            raise ValueError("Kernel centring statistics not fitted (call fit first).")
        row_mean = Kxz.mean(axis=1, keepdims=True)
        return Kxz - row_mean - self._K_col_mean_ + self._K_mean_

    def _ovr_sign_targets(self, y: np.ndarray) -> np.ndarray:
        """Return OvR targets in {-1, +1} of shape (n, C)."""

        n = y.shape[0]
        Y = -np.ones((n, self.n_classes), dtype=np.float32)
        Y[np.arange(n), y.astype(int)] = 1.0
        return Y

    @staticmethod
    def _mean_ovr_hinge(scores: np.ndarray, Y: np.ndarray) -> float:
        """Mean OvR hinge loss: mean(max(0, 1 - y*s)) over (n, C)."""

        return float(np.maximum(0.0, 1.0 - Y * scores).mean())

    def fit(
        self,
        X: Optional[np.ndarray],
        y: np.ndarray,
        *,
        K: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        K_val: Optional[np.ndarray] = None,
    ) -> "KernelSVM":
        """Fit the model.

        Args:
            X: Training features of shape (n, d).
            y: Integer labels of shape (n,).
            K: Optional precomputed training Gram matrix of shape (n, n). Use
                this when kernel_fn is None.
            X_val: Optional validation features of shape (m, d).
            y_val: Optional validation labels of shape (m,).
            K_val: Optional precomputed validation cross-Gram matrix of shape
                (m, n). Use this when kernel_fn is None.

        Returns:
            Self.
        """

        if self.lam <= 0.0:
            raise ValueError("lam must be positive")

        y_arr = np.asarray(y, dtype=int)
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array")

        if self.kernel_fn is None:
            if K is None:
                raise ValueError("kernel_fn is None; provide precomputed K in fit(..., K=...)")
            K = np.asarray(K, dtype=np.float32)
            if K.ndim != 2 or K.shape[0] != K.shape[1]:
                raise ValueError("K must be a square (n, n) matrix")
            if K.shape[0] != y_arr.shape[0]:
                raise ValueError("K must have shape (n, n) where n=len(y)")

            # In precomputed mode we don't need features; keep a tiny placeholder
            # so predict() can validate shapes consistently.
            self.X_train_ = np.empty((int(K.shape[0]), 0), dtype=np.float32)
        else:
            if X is None:
                raise ValueError("X is required when kernel_fn is provided")
            X_arr = np.asarray(X, dtype=np.float32)
            if X_arr.ndim != 2:
                raise ValueError("X must be a 2D array")
            if y_arr.shape[0] != X_arr.shape[0]:
                raise ValueError("y must be a 1D array with the same length as X")
            self.X_train_ = X_arr

            K = self.kernel_fn(X_arr, X_arr)
            K = np.asarray(K, dtype=np.float32)
            if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] != X_arr.shape[0]:
                raise ValueError("K must have shape (n, n) where n=len(X)")

        K = self._centre_train_gram(K)

        n = K.shape[0]
        Cbox = 1.0 / (2.0 * self.lam * float(n))
        Y = self._ovr_sign_targets(y_arr)  # (n, C)

        # Validation cache.
        K_val_cache: Optional[np.ndarray] = None
        Y_val: Optional[np.ndarray] = None
        if y_val is not None:
            yv = np.asarray(y_val, dtype=int)
            if yv.ndim != 1:
                raise ValueError("y_val must be a 1D array")

            if self.kernel_fn is None:
                if K_val is None:
                    raise ValueError("kernel_fn is None; provide precomputed K_val when using validation")
                K_val_cache = np.asarray(K_val, dtype=np.float32)
            else:
                if X_val is None:
                    raise ValueError("X_val is required when kernel_fn is provided")
                Xv = np.asarray(X_val, dtype=np.float32)
                if Xv.ndim != 2:
                    raise ValueError("X_val must be a 2D array")
                K_val_cache = self.kernel_fn(Xv, self.X_train_)
                K_val_cache = np.asarray(K_val_cache, dtype=np.float32)

            if K_val_cache.ndim != 2 or K_val_cache.shape[0] != yv.shape[0] or K_val_cache.shape[1] != n:
                raise ValueError("K_val must have shape (m, n) for validation")

            K_val_cache = self._centre_cross_gram(K_val_cache)
            Y_val = self._ovr_sign_targets(yv)

        # Safe step scaling: ||K||_2 <= max_i sum_j |K_ij|.
        max_row_sum = float(np.abs(K).sum(axis=1).max())
        step = float(self.lr) / (2.0 * max_row_sum + 1e-12)

        A = np.zeros((n, self.n_classes), dtype=np.float32)
        best_A = A.copy()
        best_val = float("inf")
        bad = 0

        for _ in range(self.epochs):
            KA = K @ A
            # Gradient ascent: A <- A + 2*step*(Y - K@A)
            A = A + (2.0 * step) * (Y - KA)

            # Project onto 0 <= y*a <= Cbox.
            AY = A * Y
            AY = np.clip(AY, 0.0, Cbox)
            A = AY * Y

            if K_val_cache is not None and Y_val is not None:
                scores_val = K_val_cache @ A
                hinge = self._mean_ovr_hinge(scores_val, Y_val)
                # RKHS norm per class: sum_k a_k^T K a_k = trace(A^T K A)
                KA = K @ A
                rkhs_norm = float((A * KA).sum())
                val_loss = hinge + self.lam * rkhs_norm

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_A = A.copy()
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

        self.A_ = best_A if (K_val_cache is not None and Y_val is not None) else A
        return self

    def predict(self, X: Optional[np.ndarray] = None, *, K_star: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and return OvR scores.

        Args:
            X: Features of shape (m, d). Required when kernel_fn is not None.
            K_star: Optional precomputed cross-Gram matrix of shape (m, n).
                Required when kernel_fn is None.

        Returns:
            A tuple (y_pred, scores) where:
            - y_pred has shape (m,) with integer class predictions.
            - scores has shape (m, C) with OvR decision scores.
        """

        if self.A_ is None or self.X_train_ is None:
            raise ValueError("Call fit() before predict().")

        if self.kernel_fn is None:
            if K_star is None:
                raise ValueError("kernel_fn is None; provide K_star in predict(..., K_star=...)")
            K_star = np.asarray(K_star, dtype=np.float32)
        else:
            if X is None:
                raise ValueError("X is required when kernel_fn is provided")
            X_arr = np.asarray(X, dtype=np.float32)
            K_star = self.kernel_fn(X_arr, self.X_train_)
            K_star = np.asarray(K_star, dtype=np.float32)

        if K_star.ndim != 2 or K_star.shape[1] != self.X_train_.shape[0]:
            raise ValueError("K_star must have shape (m, n) where n=len(X_train)")
        K_star = self._centre_cross_gram(K_star)
        scores = K_star @ self.A_
        return scores.argmax(axis=1), scores