from __future__ import annotations

from typing import Callable, Optional

import numpy as np


KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
DiagFn = Callable[[np.ndarray], np.ndarray]


def lin_diag(Z: np.ndarray) -> np.ndarray:
    """Diagonal for the linear kernel: K(x,x)=||x||^2."""

    Z = np.asarray(Z, dtype=np.float32)
    return np.sum(Z * Z, axis=1).astype(np.float32)


def unit_diag(Z: np.ndarray) -> np.ndarray:
    """Diagonal for kernels with K(x,x)=1 (e.g., RBF/Laplacian/chi2-RBF)."""

    return np.ones((np.asarray(Z).shape[0],), dtype=np.float32)


def gram_diag_from_kernel_fn(
    kernel_fn: KernelFn,
    Z: np.ndarray,
    *,
    block_size: int = 512,
) -> np.ndarray:
    """Compute diag(kernel_fn(Z,Z)) without forming the full Gram in memory.

    Note: this still does O(n^2) kernel work in general; provide a fast diag_fn
    (e.g., unit_diag) when available.
    """

    Z = np.asarray(Z, dtype=np.float32)
    n = int(Z.shape[0])
    out = np.empty((n,), dtype=np.float32)

    bs = int(block_size)
    if bs <= 0:
        raise ValueError("block_size must be positive")

    for start in range(0, n, bs):
        stop = min(n, start + bs)
        Kbb = np.asarray(kernel_fn(Z[start:stop], Z[start:stop]), dtype=np.float32)
        out[start:stop] = np.diag(Kbb)

    return out


def normalise_train_gram(K: np.ndarray, diag: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Unit-diagonal normalisation: K_ij /= sqrt(d_i d_j)."""

    K = np.asarray(K, dtype=np.float32)
    d = np.asarray(diag, dtype=np.float32)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be square")
    if d.ndim != 1 or d.shape[0] != K.shape[0]:
        raise ValueError("diag must have shape (n,)")

    d = np.maximum(d, float(eps))
    return (K / np.sqrt(d[:, None] * d[None, :])).astype(np.float32)


def normalise_cross_gram(
    Kxz: np.ndarray,
    diag_x: np.ndarray,
    diag_z: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Cross-set unit-diagonal normalisation: Kxz_ij /= sqrt(d_x[i] d_z[j])."""

    Kxz = np.asarray(Kxz, dtype=np.float32)
    dx = np.asarray(diag_x, dtype=np.float32)
    dz = np.asarray(diag_z, dtype=np.float32)

    if Kxz.ndim != 2:
        raise ValueError("Kxz must be 2D")
    if dx.ndim != 1 or dx.shape[0] != Kxz.shape[0]:
        raise ValueError("diag_x must have shape (m,)")
    if dz.ndim != 1 or dz.shape[0] != Kxz.shape[1]:
        raise ValueError("diag_z must have shape (n,)")

    dx = np.maximum(dx, float(eps))
    dz = np.maximum(dz, float(eps))
    return (Kxz / np.sqrt(dx[:, None] * dz[None, :])).astype(np.float32)


def resolve_diag_fn(kernel_fn: KernelFn, diag_fn: Optional[DiagFn]) -> DiagFn:
    """Choose a diagonal function; fallback computes diag via kernel_fn."""

    if diag_fn is not None:
        return diag_fn

    def _fallback(Z: np.ndarray) -> np.ndarray:
        return gram_diag_from_kernel_fn(kernel_fn, Z)

    return _fallback
