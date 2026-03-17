from typing import Callable, Optional, Tuple

import numpy as np


KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class KernelRidgeRegression:
    """Multiclass kernel ridge regression (KRR) in the dual.

    Learns dual coefficients A over the training set such that the score for class k is:
        s_k(x) = sum_i A_{ik} K(x_i, x)

    With one-hot targets Y (n, C), the standard KRR solution is:
        A = (K + n * lam * I)^{-1} Y

    This implementation uses a bias-free model by centring the Gram matrix (as in the
    other kernel models in this repo) and reuses the same centring statistics at predict time.
    """

    def __init__(
        self,
        n_classes: int,
        *,
        kernel_fn: Optional[KernelFn] = None,
        lam: float = 1e-4,
        jitter: float = 1e-8,
        solver: str = "cg",
        cg_tol: float = 1e-6,
        cg_max_iter: Optional[int] = None,
    ):
        """Create a kernel ridge regression estimator.

        Args:
            n_classes: Number of classes.
            kernel_fn: Function returning the Gram matrix K(X, Z). If None, you
                must pass precomputed kernels via fit(..., K=...) and
                predict(..., K_star=...).
            lam: Regularisation strength (appears as n*lam in the dual system).
            jitter: Small diagonal stabiliser added on top of the ridge term.
            solver: Linear solver to use: "cg" (iterative) or "direct".
            cg_tol: Relative tolerance for CG (smaller is more accurate but slower).
            cg_max_iter: Maximum CG iterations. If None, uses 5*n.
        """

        self.n_classes = int(n_classes)
        self.kernel_fn = kernel_fn
        self.lam = float(lam)
        self.jitter = float(jitter)

        self.solver = str(solver).lower()
        self.cg_tol = float(cg_tol)
        self.cg_max_iter = None if cg_max_iter is None else int(cg_max_iter)

        self.X_train_: Optional[np.ndarray] = None
        self.A_: Optional[np.ndarray] = None  # (n, C)

        self._K_col_mean_: Optional[np.ndarray] = None
        self._K_mean_: Optional[float] = None

    @staticmethod
    def _cg_solve(
        A_mv: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
        *,
        tol: float,
        max_iter: int,
    ) -> np.ndarray:
        """Conjugate gradient solve for SPD systems: A x = b.

        Args:
            A_mv: Function computing A @ v.
            b: Right-hand side (n,).
            tol: Relative residual tolerance.
            max_iter: Max iterations.

        Returns:
            Solution vector x (n,).
        """

        b = np.asarray(b)
        if b.ndim != 1:
            raise ValueError("b must be a 1D array")

        # Start from x0 = 0.
        x = np.zeros_like(b, dtype=np.float64)
        r = b.astype(np.float64, copy=False) - A_mv(x.astype(b.dtype, copy=False)).astype(
            np.float64, copy=False
        )
        p = r.copy()
        rs0 = float(r @ r)
        if rs0 <= 0.0:
            return x.astype(np.float32, copy=False)

        rs = rs0
        atol = (tol * np.sqrt(rs0))

        for _ in range(int(max_iter)):
            Ap = A_mv(p.astype(b.dtype, copy=False)).astype(np.float64, copy=False)
            denom = float(p @ Ap)
            if denom <= 1e-30:
                break
            alpha = rs / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = float(r @ r)
            if np.sqrt(rs_new) <= atol:
                break
            beta = rs_new / (rs + 1e-30)
            p = r + beta * p
            rs = rs_new

        return x.astype(np.float32, copy=False)

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
        row_mean = Kxz.mean(axis=1, keepdims=True)  # (m, 1)
        return Kxz - row_mean - self._K_col_mean_ + self._K_mean_

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        Y = np.zeros((y.shape[0], self.n_classes), dtype=np.float32)
        Y[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return Y

    def fit(
        self,
        X: Optional[np.ndarray],
        y: np.ndarray,
        *,
        K: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        K_val: Optional[np.ndarray] = None,
    ) -> "KernelRidgeRegression":
        """Fit the model.

        Args:
            X: Training features of shape (n, d).
            y: Integer labels of shape (n,).
            K: Optional precomputed training Gram matrix of shape (n, n). Use
                this when kernel_fn is None.
            X_val: Unused (accepted for API compatibility).
            y_val: Unused (accepted for API compatibility).
            K_val: Unused (accepted for API compatibility).

        Returns:
            Self.
        """

        if self.lam <= 0.0:
            raise ValueError("lam must be positive")
        if self.jitter < 0.0:
            raise ValueError("jitter must be non-negative")

        if self.kernel_fn is None and K is None:
            raise ValueError("kernel_fn is None; provide precomputed K in fit(..., K=...)")

        if self.solver not in {"cg", "direct"}:
            raise ValueError('solver must be either "cg" or "direct"')
        if self.cg_tol <= 0.0:
            raise ValueError("cg_tol must be positive")

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

            # Precomputed mode: features are not needed.
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

        Y = self._one_hot(y_arr)
        n = int(K.shape[0])

        reg = (n * self.lam) + self.jitter

        if self.solver == "direct":
            K_reg = K + (reg * np.eye(n, dtype=np.float32))
            A = np.linalg.solve(K_reg.astype(np.float64), Y.astype(np.float64))
            self.A_ = A.astype(np.float32, copy=False)
            return self

        # Iterative CG solve using matrix-vector products with (K + reg I).
        def A_mv(v: np.ndarray) -> np.ndarray:
            v_arr = np.asarray(v, dtype=np.float32)
            return (K @ v_arr) + (reg * v_arr)

        max_iter = int(5 * n) if self.cg_max_iter is None else int(self.cg_max_iter)
        A = np.empty((n, self.n_classes), dtype=np.float32)
        for k in range(self.n_classes):
            A[:, k] = self._cg_solve(A_mv, Y[:, k], tol=self.cg_tol, max_iter=max_iter)

        self.A_ = A
        return self

    def predict(self, X: Optional[np.ndarray] = None, *, K_star: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and return KRR scores.

        Args:
            X: Features of shape (m, d). Required when kernel_fn is not None.
            K_star: Optional precomputed cross-Gram matrix of shape (m, n).
                Required when kernel_fn is None.

        Returns:
            A tuple (y_pred, scores) where:
            - y_pred has shape (m,) with integer class predictions.
            - scores has shape (m, C) with KRR scores.
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
