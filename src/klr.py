from typing import Callable, Optional, Tuple

import numpy as np


KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class KernelLogisticRegression:
    """Multiclass kernel logistic regression with gradient descent.

    This model learns dual coefficients over the training set and supports
    optional early stopping using a validation set.
    """

    def __init__(
                self,
                n_classes: int,
                *,
                kernel_fn: KernelFn,
                lr: float = 0.1,
                epochs: int = 200,
                lam: float = 1e-4,
                patience: int = 10,
        ):
        """Create a kernel logistic regression estimator.

        Args:
            n_classes: Number of classes.
            kernel_fn: Function returning the Gram matrix K(X, Z).
            lr: Gradient descent learning rate.
            epochs: Maximum number of optimisation steps.
            lam: L2 regularisation strength (applied in RKHS via K @ A term).
            patience: Early stopping patience (in epochs) when validation is used.
        """
        self.n_classes = int(n_classes)
        self.kernel_fn = kernel_fn
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.lam = float(lam)
        self.patience = int(patience)

        self.X_train_: Optional[np.ndarray] = None
        self.A_: Optional[np.ndarray] = None  # (n, C)

        self._K_col_mean_: Optional[np.ndarray] = None
        self._K_mean_: Optional[float] = None

    def _centre_train_gram(self, K: np.ndarray) -> np.ndarray:
        """Centre a square training Gram matrix.

        Stores centring statistics for later use on cross-Gram matrices.

        Args:
            K: Training Gram matrix of shape (n, n).

        Returns:
            Centred Gram matrix of shape (n, n).
        """
        row_mean = K.mean(axis=1, keepdims=True)  # (n, 1)
        col_mean = K.mean(axis=0, keepdims=True)  # (1, n)
        mean = float(K.mean())

        self._K_col_mean_ = col_mean
        self._K_mean_ = mean

        return K - row_mean - col_mean + mean

    def _centre_cross_gram(self, Kxz: np.ndarray) -> np.ndarray:
        """Centre a cross-Gram matrix using fitted training statistics.

        Args:
            Kxz: Cross-Gram matrix of shape (m, n) where n is the training size.

        Returns:
            Centred cross-Gram matrix of shape (m, n).
        """
        if self._K_col_mean_ is None or self._K_mean_ is None:
            raise ValueError("Kernel centring statistics not fitted (call fit first).")
        row_mean = Kxz.mean(axis=1, keepdims=True)  # (m, 1)
        return Kxz - row_mean - self._K_col_mean_ + self._K_mean_

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        """Convert integer class labels to a one-hot encoded matrix."""

        Y = np.zeros((y.shape[0], self.n_classes), dtype=np.float32)
        Y[np.arange(y.shape[0]), y.astype(int)] = 1.0
        return Y

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        """Row-wise softmax with basic numerical stabilisation."""

        Zs = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Zs)
        return expZ / (expZ.sum(axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def _cross_entropy(Y: np.ndarray, P: np.ndarray) -> float:
        """Mean multiclass cross-entropy (log loss)."""

        return float(-(Y * np.log(P + 1e-12)).sum(axis=1).mean())

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "KernelLogisticRegression":
        """Fit the model.

        Args:
            X: Training features of shape (n, d).
            y: Integer labels of shape (n,).
            X_val: Optional validation features of shape (m, d).
            y_val: Optional validation labels of shape (m,).

        Returns:
            Self.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        self.X_train_ = X

        K = self.kernel_fn(X, X)
        K = np.asarray(K, dtype=np.float32)
        K = self._centre_train_gram(K)

        Y = self._one_hot(y)

        K_val: Optional[np.ndarray] = None
        Y_val: Optional[np.ndarray] = None
        if X_val is not None and y_val is not None:
            X_val_arr = np.asarray(X_val, dtype=np.float32)
            y_val_arr = np.asarray(y_val, dtype=int)
            K_val = self.kernel_fn(X_val_arr, X)
            K_val = np.asarray(K_val, dtype=np.float32)
            K_val = self._centre_cross_gram(K_val)
            Y_val = self._one_hot(y_val_arr)

        n = K.shape[0]
        A = np.zeros((n, self.n_classes), dtype=np.float32)

        best_A = A.copy()
        best_val = float("inf")
        bad = 0

        for _ in range(self.epochs):
            Z = K @ A
            P = self._softmax(Z)

            grad = (K @ (P - Y)) / n + self.lam * (K @ A)
            A = A - self.lr * grad
            
            if K_val is not None and Y_val is not None:
                Pv = self._softmax(K_val @ A)
                val_loss = self._cross_entropy(Y_val, Pv)
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_A = A.copy()
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

        self.A_ = best_A if (K_val is not None and Y_val is not None) else A
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and probabilities.

        Args:
            X: Features of shape (m, d).

        Returns:
            A tuple (y_pred, P) where:
            - y_pred has shape (m,) with integer class predictions.
            - P has shape (m, C) with class probabilities.
        """
        if self.A_ is None or self.X_train_ is None:
            raise ValueError("Call fit() before predict().")

        X_arr = np.asarray(X, dtype=np.float32)
        K_star = self.kernel_fn(X_arr, self.X_train_)
        K_star = np.asarray(K_star, dtype=np.float32)
        K_star = self._centre_cross_gram(K_star)

        P = self._softmax(K_star @ self.A_)
        return P.argmax(axis=1), P
