from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, NotRequired, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np

from src.cv import fold_train_val_indices, make_folds
from src.kernel_normalisation import (
    DiagFn,
    KernelFn,
    normalise_cross_gram,
    normalise_train_gram,
    resolve_diag_fn,
)
from src.krr import KernelRidgeRegression
from src.metrics import accuracy
from src.svm import KernelSVM


class KernelSpec(TypedDict):
    """Config for one feature-space kernel.

    Required keys:
        Z: Feature matrix for all samples (n, d).
        kernel_fn: Function computing the Gram matrix K(Za, Zb).

    Optional keys:
        name: Used for caching; should be unique per spec.
        diag_fn: Fast diagonal function for normalisation.
        normalise: Whether to apply unit-diagonal normalisation (default True).
    """

    Z: np.ndarray
    kernel_fn: KernelFn
    name: NotRequired[str]
    diag_fn: NotRequired[DiagFn]
    normalise: NotRequired[bool]


@dataclass
class KernelFoldCache:
    K_tr: np.ndarray
    K_va_tr: np.ndarray


def beta_grid(step: float = 0.1, *, include_ends: bool = True) -> np.ndarray:
    """Grid of betas in [0, 1]."""

    if step <= 0:
        raise ValueError("step must be positive")

    if include_ends:
        # Include 0 and 1 robustly.
        n = int(np.floor(1.0 / step + 1e-12))
        grid = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
        grid[0] = 0.0
        grid[-1] = 1.0
        return grid

    grid = np.arange(step, 1.0, step, dtype=np.float32)
    return grid[(grid > 0.0) & (grid < 1.0)]


def _validate_kernel_specs_two(specs: Sequence[KernelSpec]) -> Tuple[KernelSpec, KernelSpec]:
    if len(specs) != 2:
        raise ValueError("This implementation currently supports exactly 2 kernels")

    a, b = specs[0], specs[1]
    for s in (a, b):
        if "Z" not in s or "kernel_fn" not in s:
            raise ValueError("Each KernelSpec must contain at least 'Z' and 'kernel_fn'")

    Za = np.asarray(a["Z"])
    Zb = np.asarray(b["Z"])
    if Za.shape[0] != Zb.shape[0]:
        raise ValueError("All specs must have the same number of samples")

    return a, b


def _validate_kernel_specs_three(specs: Sequence[KernelSpec]) -> Tuple[KernelSpec, KernelSpec, KernelSpec]:
    if len(specs) != 3:
        raise ValueError("This implementation currently supports exactly 3 kernels")

    a, b, c = specs[0], specs[1], specs[2]
    for s in (a, b, c):
        if "Z" not in s or "kernel_fn" not in s:
            raise ValueError("Each KernelSpec must contain at least 'Z' and 'kernel_fn'")

    Za = np.asarray(a["Z"])
    Zb = np.asarray(b["Z"])
    Zc = np.asarray(c["Z"])
    if Za.shape[0] != Zb.shape[0] or Za.shape[0] != Zc.shape[0]:
        raise ValueError("All specs must have the same number of samples")

    return a, b, c


def _validate_kernel_specs_four(specs: Sequence[KernelSpec]) -> Tuple[KernelSpec, KernelSpec, KernelSpec, KernelSpec]:
    if len(specs) != 4:
        raise ValueError("This implementation currently supports exactly 4 kernels")

    a, b, c, d = specs[0], specs[1], specs[2], specs[3]
    for s in (a, b, c, d):
        if "Z" not in s or "kernel_fn" not in s:
            raise ValueError("Each KernelSpec must contain at least 'Z' and 'kernel_fn'")

    Za = np.asarray(a["Z"])
    Zb = np.asarray(b["Z"])
    Zc = np.asarray(c["Z"])
    Zd = np.asarray(d["Z"])
    n = int(Za.shape[0])
    if int(Zb.shape[0]) != n or int(Zc.shape[0]) != n or int(Zd.shape[0]) != n:
        raise ValueError("All specs must have the same number of samples")

    return a, b, c, d


def _iter_w12_pairs(
    w12_grid: Union[np.ndarray, Sequence[Tuple[float, float]], Tuple[Sequence[float], Sequence[float]]],
) -> Iterable[Tuple[float, float]]:
    """Yield (w1, w2) pairs.

    Supported formats:
        - Array-like of shape (m, 2)
        - Sequence of (w1, w2) pairs
        - Tuple (w1_values, w2_values) interpreted as a cartesian product
    """

    if isinstance(w12_grid, tuple) and len(w12_grid) == 2:
        w1_vals, w2_vals = w12_grid
        for w1 in w1_vals:
            for w2 in w2_vals:
                yield float(w1), float(w2)
        return

    arr = np.asarray(w12_grid, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("w12_grid must be shape (m, 2), a list of pairs, or (w1s, w2s)")

    for i in range(int(arr.shape[0])):
        yield float(arr[i, 0]), float(arr[i, 1])


def _iter_w123_triples(
    w123_grid: Union[
        np.ndarray,
        Sequence[Tuple[float, float, float]],
        Tuple[Sequence[float], Sequence[float], Sequence[float]],
    ],
) -> Iterable[Tuple[float, float, float]]:
    """Yield (w1, w2, w3) triples.

    Supported formats:
        - Array-like of shape (m, 3)
        - Sequence of (w1, w2, w3) triples
        - Tuple (w1_values, w2_values, w3_values) interpreted as a cartesian product
    """

    if isinstance(w123_grid, tuple) and len(w123_grid) == 3:
        w1_vals, w2_vals, w3_vals = w123_grid
        for w1 in w1_vals:
            for w2 in w2_vals:
                for w3 in w3_vals:
                    yield float(w1), float(w2), float(w3)
        return

    arr = np.asarray(w123_grid, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("w123_grid must be shape (m, 3), a list of triples, or (w1s, w2s, w3s)")

    for i in range(int(arr.shape[0])):
        yield float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])


def _compute_fold_grams_for_spec(
    spec: KernelSpec,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    *,
    eps: float,
) -> KernelFoldCache:
    Z = np.asarray(spec["Z"], dtype=np.float32)
    kernel_fn = spec["kernel_fn"]
    diag_fn = resolve_diag_fn(kernel_fn, spec.get("diag_fn"))
    do_norm = bool(spec.get("normalise", True))

    Z_tr = Z[tr_idx]
    Z_va = Z[va_idx]

    K_tr = np.asarray(kernel_fn(Z_tr, Z_tr), dtype=np.float32)
    K_va_tr = np.asarray(kernel_fn(Z_va, Z_tr), dtype=np.float32)

    if not do_norm:
        return KernelFoldCache(K_tr=K_tr, K_va_tr=K_va_tr)

    d_tr = np.asarray(diag_fn(Z_tr), dtype=np.float32)
    d_va = np.asarray(diag_fn(Z_va), dtype=np.float32)

    K_tr = normalise_train_gram(K_tr, d_tr, eps=eps)
    K_va_tr = normalise_cross_gram(K_va_tr, d_va, d_tr, eps=eps)

    return KernelFoldCache(K_tr=K_tr, K_va_tr=K_va_tr)


def _cached_fold_grams(
    cache: Dict[Tuple[int, str], KernelFoldCache],
    *,
    fold: int,
    spec: KernelSpec,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    eps: float,
) -> KernelFoldCache:
    name = str(spec.get("name", "kernel"))
    key = (int(fold), name)

    hit = cache.get(key)
    if hit is not None:
        return hit

    grams = _compute_fold_grams_for_spec(spec, tr_idx, va_idx, eps=eps)
    cache[key] = grams
    return grams


def _combine_two_kernels(
    Ka: KernelFoldCache,
    Kb: KernelFoldCache,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    b = float(beta)
    if not (0.0 <= b <= 1.0):
        raise ValueError("beta must be in [0, 1]")

    Ktr = (b * Ka.K_tr) + ((1.0 - b) * Kb.K_tr)
    Kva = (b * Ka.K_va_tr) + ((1.0 - b) * Kb.K_va_tr)
    return Ktr.astype(np.float32, copy=False), Kva.astype(np.float32, copy=False)


def beta_grid_search_cv_two_kernels(
    specs: Sequence[KernelSpec],
    y: np.ndarray,
    *,
    n_classes: int,
    model: str,
    betas: Union[np.ndarray, Sequence[float]] = (0.0, 0.25, 0.5, 0.75, 1.0),
    k: int = 5,
    seed: int = 0,
    normalisation_eps: float = 1e-12,
    # SVM params
    lam: float = 1e-5,
    lr: float = 0.6,
    epochs: int = 500,
    patience: int = 5,
    # KRR params
    krr_solver: str = "cg",
    krr_cg_tol: float = 1e-6,
    # Reporting
    verbose: int = 0,
    progress_cb: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    """K-fold CV over beta for a 2-kernel convex combination.

    Uses unit-diagonal normalisation per-kernel per-fold (unless spec["normalise"]=False).

    Args:
        specs: Exactly two KernelSpec dicts.
        y: Labels (n,).
        n_classes: Number of classes.
        model: "svm" or "krr".
        betas: Sequence of betas in [0,1] where final mix is beta*K1 + (1-beta)*K2.
        k: Number of CV folds.
        seed: RNG seed for folds.
        normalisation_eps: Epsilon in normalisation.

    Returns:
        Dict with keys:
            - "best_beta"
            - "best_mean_acc"
            - "mean_accs" (dict beta->mean acc)
            - "fold_accs" (dict beta->list of fold accs)

    Reporting:
        Set verbose=1 to print one line per beta, verbose=2 to print per-fold lines.
        Alternatively (or additionally), supply progress_cb(event_dict) to receive
        streaming progress events.
    """

    spec_a, spec_b = _validate_kernel_specs_two(specs)

    y = np.asarray(y, dtype=int)
    n = int(y.shape[0])

    folds = make_folds(n, k=int(k), seed=int(seed))
    betas_arr = np.asarray(list(betas), dtype=np.float32)

    model_key = str(model).lower().strip()
    if model_key not in {"svm", "krr"}:
        raise ValueError('model must be "svm" or "krr"')

    # Cache individual kernel matrices per fold/spec.
    grams_cache: Dict[Tuple[int, str], KernelFoldCache] = {}

    fold_accs: Dict[float, List[float]] = {}
    mean_accs: Dict[float, float] = {}

    def _emit(event: Dict[str, object]) -> None:
        if progress_cb is not None:
            progress_cb(event)

        v = int(verbose)
        if v <= 0:
            return

        et = event.get("type")
        if et == "beta_end" and v >= 1:
            beta = float(event["beta"])  # type: ignore[assignment]
            mean_acc = float(event["mean_acc"])  # type: ignore[assignment]
            print(f"beta={beta:.3f} mean_acc={mean_acc:.4f}")
        elif et == "fold_end" and v >= 2:
            beta = float(event["beta"])  # type: ignore[assignment]
            fold = int(event["fold"])  # type: ignore[assignment]
            n_folds = int(event["n_folds"])  # type: ignore[assignment]
            acc = float(event["acc"])  # type: ignore[assignment]
            mean_so_far = float(event["mean_so_far"])  # type: ignore[assignment]
            print(f"beta={beta:.3f} fold={fold+1}/{n_folds} acc={acc:.4f} mean_so_far={mean_so_far:.4f}")

    for beta in betas_arr:
        b = float(beta)
        accs: List[float] = []

        _emit({"type": "beta_start", "beta": b})

        for f in range(len(folds)):
            tr_idx, va_idx = fold_train_val_indices(folds, f)

            Ka = _cached_fold_grams(
                grams_cache,
                fold=f,
                spec=spec_a,
                tr_idx=tr_idx,
                va_idx=va_idx,
                eps=float(normalisation_eps),
            )
            Kb = _cached_fold_grams(
                grams_cache,
                fold=f,
                spec=spec_b,
                tr_idx=tr_idx,
                va_idx=va_idx,
                eps=float(normalisation_eps),
            )

            Ktr, Kva_tr = _combine_two_kernels(Ka, Kb, b)

            y_tr, y_va = y[tr_idx], y[va_idx]

            if model_key == "svm":
                clf = KernelSVM(
                    n_classes=int(n_classes),
                    kernel_fn=None,
                    lam=float(lam),
                    lr=float(lr),
                    epochs=int(epochs),
                    patience=int(patience),
                ).fit(
                    None,
                    y_tr,
                    K=Ktr,
                    y_val=y_va,
                    K_val=Kva_tr,
                )
                pred_va, _ = clf.predict(K_star=Kva_tr)
            else:
                clf = KernelRidgeRegression(
                    n_classes=int(n_classes),
                    kernel_fn=None,
                    lam=float(lam),
                    solver=str(krr_solver),
                    cg_tol=float(krr_cg_tol),
                ).fit(
                    None,
                    y_tr,
                    K=Ktr,
                )
                pred_va, _ = clf.predict(K_star=Kva_tr)

            accs.append(float(accuracy(y_va, pred_va)))

            _emit(
                {
                    "type": "fold_end",
                    "beta": b,
                    "fold": int(f),
                    "n_folds": int(len(folds)),
                    "acc": float(accs[-1]),
                    "mean_so_far": float(np.mean(accs)),
                }
            )

        fold_accs[b] = accs
        mean_accs[b] = float(np.mean(accs))

        _emit({"type": "beta_end", "beta": b, "mean_acc": float(mean_accs[b]), "fold_accs": accs})

    best_beta = max(mean_accs, key=lambda bb: mean_accs[bb])

    _emit(
        {
            "type": "search_end",
            "best_beta": float(best_beta),
            "best_mean_acc": float(mean_accs[best_beta]),
        }
    )

    return {
        "best_beta": float(best_beta),
        "best_mean_acc": float(mean_accs[best_beta]),
        "mean_accs": mean_accs,
        "fold_accs": fold_accs,
    }


def weight_grid_search_cv_three_kernels(
    specs: Sequence[KernelSpec],
    y: np.ndarray,
    *,
    n_classes: int,
    model: str,
    w12_grid: Union[np.ndarray, Sequence[Tuple[float, float]], Tuple[Sequence[float], Sequence[float]]],
    k: int = 5,
    seed: int = 0,
    normalisation_eps: float = 1e-12,
    # SVM params
    lam: float = 1e-4,
    lr: float = 0.6,
    epochs: int = 500,
    patience: int = 5,
    # KRR params
    krr_solver: str = "cg",
    krr_cg_tol: float = 1e-6,
    # Reporting
    verbose: int = 0,
    progress_cb: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    """K-fold CV over 3-kernel convex weights.

    Weights are (w1, w2, w3) with w3 = 1 - w1 - w2 and all weights >= 0.
    The combined kernel is K = w1*K1 + w2*K2 + w3*K3.

    Each kernel is computed per-fold and can be unit-diagonal normalised per-kernel per-fold
    (unless spec["normalise"]=False), matching the behaviour of `beta_grid_search_cv_two_kernels`.

    Args:
        specs: Exactly three KernelSpec dicts.
        y: Labels (n,).
        n_classes: Number of classes.
        model: "svm" or "krr".
        w12_grid: Candidate pairs (w1, w2). Supports:
            - array-like of shape (m, 2)
            - list of (w1, w2) pairs
            - tuple (w1_values, w2_values) interpreted as cartesian product.
        k: Number of CV folds.
        seed: RNG seed for folds.
        normalisation_eps: Epsilon in normalisation.

    Returns:
        Dict with keys:
            - "best_w" (tuple (w1,w2,w3))
            - "best_mean_acc"
            - "mean_accs" (dict (w1,w2,w3)->mean acc)
            - "fold_accs" (dict (w1,w2,w3)->list of fold accs)
    """

    spec1, spec2, spec3 = _validate_kernel_specs_three(specs)

    y = np.asarray(y, dtype=int)
    n = int(y.shape[0])
    folds = make_folds(n, k=int(k), seed=int(seed))

    model_key = str(model).lower().strip()
    if model_key not in {"svm", "krr"}:
        raise ValueError('model must be "svm" or "krr"')

    grams_cache: Dict[Tuple[int, str], KernelFoldCache] = {}

    fold_accs: Dict[Tuple[float, float, float], List[float]] = {}
    mean_accs: Dict[Tuple[float, float, float], float] = {}

    tol = 1e-8

    def _emit(event: Dict[str, object]) -> None:
        if progress_cb is not None:
            progress_cb(event)

        v = int(verbose)
        if v <= 0:
            return

        et = event.get("type")
        if et == "w_end" and v >= 1:
            w1 = float(event["w1"])  # type: ignore[assignment]
            w2 = float(event["w2"])  # type: ignore[assignment]
            w3 = float(event["w3"])  # type: ignore[assignment]
            mean_acc = float(event["mean_acc"])  # type: ignore[assignment]
            print(f"w=({w1:.3f},{w2:.3f},{w3:.3f}) mean_acc={mean_acc:.4f}")
        elif et == "fold_end" and v >= 2:
            w1 = float(event["w1"])  # type: ignore[assignment]
            w2 = float(event["w2"])  # type: ignore[assignment]
            w3 = float(event["w3"])  # type: ignore[assignment]
            fold = int(event["fold"])  # type: ignore[assignment]
            n_folds = int(event["n_folds"])  # type: ignore[assignment]
            acc = float(event["acc"])  # type: ignore[assignment]
            mean_so_far = float(event["mean_so_far"])  # type: ignore[assignment]
            print(
                f"w=({w1:.3f},{w2:.3f},{w3:.3f}) fold={fold+1}/{n_folds} acc={acc:.4f} mean_so_far={mean_so_far:.4f}"
            )

    for (w1, w2) in _iter_w12_pairs(w12_grid):
        w3 = 1.0 - float(w1) - float(w2)
        if (w1 < -tol) or (w2 < -tol) or (w3 < -tol):
            continue
        # Clamp tiny negatives from float noise.
        w1c = float(max(0.0, w1))
        w2c = float(max(0.0, w2))
        w3c = float(max(0.0, w3))

        # Renormalise if the sum drifted slightly above 1 after clamping.
        s = w1c + w2c + w3c
        if s <= 0.0:
            continue
        if abs(s - 1.0) > 1e-6:
            w1c /= s
            w2c /= s
            w3c /= s

        _emit({"type": "w_start", "w1": w1c, "w2": w2c, "w3": w3c})

        accs: List[float] = []
        for f in range(len(folds)):
            tr_idx, va_idx = fold_train_val_indices(folds, f)

            K1 = _cached_fold_grams(
                grams_cache,
                fold=f,
                spec=spec1,
                tr_idx=tr_idx,
                va_idx=va_idx,
                eps=float(normalisation_eps),
            )
            K2 = _cached_fold_grams(
                grams_cache,
                fold=f,
                spec=spec2,
                tr_idx=tr_idx,
                va_idx=va_idx,
                eps=float(normalisation_eps),
            )
            K3 = _cached_fold_grams(
                grams_cache,
                fold=f,
                spec=spec3,
                tr_idx=tr_idx,
                va_idx=va_idx,
                eps=float(normalisation_eps),
            )

            Ktr = (w1c * K1.K_tr) + (w2c * K2.K_tr) + (w3c * K3.K_tr)
            Kva_tr = (w1c * K1.K_va_tr) + (w2c * K2.K_va_tr) + (w3c * K3.K_va_tr)
            Ktr = np.asarray(Ktr, dtype=np.float32)
            Kva_tr = np.asarray(Kva_tr, dtype=np.float32)

            y_tr, y_va = y[tr_idx], y[va_idx]

            if model_key == "svm":
                clf = KernelSVM(
                    n_classes=int(n_classes),
                    kernel_fn=None,
                    lam=float(lam),
                    lr=float(lr),
                    epochs=int(epochs),
                    patience=int(patience),
                ).fit(
                    None,
                    y_tr,
                    K=Ktr,
                    y_val=y_va,
                    K_val=Kva_tr,
                )
                pred_va, _ = clf.predict(K_star=Kva_tr)
            else:
                clf = KernelRidgeRegression(
                    n_classes=int(n_classes),
                    kernel_fn=None,
                    lam=float(lam),
                    solver=str(krr_solver),
                    cg_tol=float(krr_cg_tol),
                ).fit(
                    None,
                    y_tr,
                    K=Ktr,
                )
                pred_va, _ = clf.predict(K_star=Kva_tr)

            accs.append(float(accuracy(y_va, pred_va)))

            _emit(
                {
                    "type": "fold_end",
                    "w1": w1c,
                    "w2": w2c,
                    "w3": w3c,
                    "fold": int(f),
                    "n_folds": int(len(folds)),
                    "acc": float(accs[-1]),
                    "mean_so_far": float(np.mean(accs)),
                }
            )

        key = (float(w1c), float(w2c), float(w3c))
        fold_accs[key] = accs
        mean_accs[key] = float(np.mean(accs))

        _emit({"type": "w_end", "w1": w1c, "w2": w2c, "w3": w3c, "mean_acc": float(mean_accs[key]), "fold_accs": accs})

    if len(mean_accs) == 0:
        raise ValueError("No valid (w1,w2) pairs in w12_grid (must satisfy w1>=0, w2>=0, w1+w2<=1)")

    best_w = max(mean_accs, key=lambda ww: mean_accs[ww])
    _emit({"type": "search_end", "best_w": best_w, "best_mean_acc": float(mean_accs[best_w])})

    return {
        "best_w": best_w,
        "best_mean_acc": float(mean_accs[best_w]),
        "mean_accs": mean_accs,
        "fold_accs": fold_accs,
    }


def weight_grid_search_cv_four_kernels(
    specs: Sequence[KernelSpec],
    y: np.ndarray,
    *,
    n_classes: int,
    model: str,
    w123_grid: Union[
        np.ndarray,
        Sequence[Tuple[float, float, float]],
        Tuple[Sequence[float], Sequence[float], Sequence[float]],
    ],
    k: int = 5,
    seed: int = 0,
    normalisation_eps: float = 1e-12,
    # SVM params
    lam: float = 1e-4,
    lr: float = 0.6,
    epochs: int = 500,
    patience: int = 5,
    # KRR params
    krr_solver: str = "cg",
    krr_cg_tol: float = 1e-6,
    # Reporting
    verbose: int = 0,
    progress_cb: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    """K-fold CV over 4-kernel convex weights.

    Weights are (w1, w2, w3, w4) with w4 = 1 - w1 - w2 - w3 and all weights >= 0.
    The combined kernel is K = w1*K1 + w2*K2 + w3*K3 + w4*K4.

    Each kernel is computed per-fold and can be unit-diagonal normalised per-kernel per-fold
    (unless spec["normalise"]=False), matching the behaviour of `beta_grid_search_cv_two_kernels`.

    Args:
        specs: Exactly four KernelSpec dicts.
        y: Labels (n,).
        n_classes: Number of classes.
        model: "svm" or "krr".
        w123_grid: Candidate triples (w1, w2, w3). Supports:
            - array-like of shape (m, 3)
            - list of (w1, w2, w3) triples
            - tuple (w1_values, w2_values, w3_values) interpreted as cartesian product.
        k: Number of CV folds.
        seed: RNG seed for folds.
        normalisation_eps: Epsilon in normalisation.

    Returns:
        Dict with keys:
            - "best_w" (tuple (w1,w2,w3,w4))
            - "best_mean_acc"
            - "mean_accs" (dict (w1,w2,w3,w4)->mean acc)
            - "fold_accs" (dict (w1,w2,w3,w4)->list of fold accs)
    """

    spec1, spec2, spec3, spec4 = _validate_kernel_specs_four(specs)

    y = np.asarray(y, dtype=int)
    n = int(y.shape[0])
    folds = make_folds(n, k=int(k), seed=int(seed))

    model_key = str(model).lower().strip()
    if model_key not in {"svm", "krr"}:
        raise ValueError('model must be "svm" or "krr"')

    grams_cache: Dict[Tuple[int, str], KernelFoldCache] = {}

    fold_accs: Dict[Tuple[float, float, float, float], List[float]] = {}
    mean_accs: Dict[Tuple[float, float, float, float], float] = {}

    tol = 1e-8

    def _emit(event: Dict[str, object]) -> None:
        if progress_cb is not None:
            progress_cb(event)

        v = int(verbose)
        if v <= 0:
            return

        et = event.get("type")
        if et == "w_end" and v >= 1:
            w1 = float(event["w1"])  # type: ignore[assignment]
            w2 = float(event["w2"])  # type: ignore[assignment]
            w3 = float(event["w3"])  # type: ignore[assignment]
            w4 = float(event["w4"])  # type: ignore[assignment]
            mean_acc = float(event["mean_acc"])  # type: ignore[assignment]
            print(f"w=({w1:.3f},{w2:.3f},{w3:.3f},{w4:.3f}) mean_acc={mean_acc:.4f}")
        elif et == "fold_end" and v >= 2:
            w1 = float(event["w1"])  # type: ignore[assignment]
            w2 = float(event["w2"])  # type: ignore[assignment]
            w3 = float(event["w3"])  # type: ignore[assignment]
            w4 = float(event["w4"])  # type: ignore[assignment]
            fold = int(event["fold"])  # type: ignore[assignment]
            n_folds = int(event["n_folds"])  # type: ignore[assignment]
            acc = float(event["acc"])  # type: ignore[assignment]
            mean_so_far = float(event["mean_so_far"])  # type: ignore[assignment]
            print(
                f"w=({w1:.3f},{w2:.3f},{w3:.3f},{w4:.3f}) fold={fold+1}/{n_folds} acc={acc:.4f} mean_so_far={mean_so_far:.4f}"
            )

    for (w1, w2, w3) in _iter_w123_triples(w123_grid):
        w4 = 1.0 - float(w1) - float(w2) - float(w3)
        if (w1 < -tol) or (w2 < -tol) or (w3 < -tol) or (w4 < -tol):
            continue

        # Clamp tiny negatives from float noise.
        w1c = float(max(0.0, w1))
        w2c = float(max(0.0, w2))
        w3c = float(max(0.0, w3))
        w4c = float(max(0.0, w4))

        # Renormalise if the sum drifted slightly above 1 after clamping.
        s = w1c + w2c + w3c + w4c
        if s <= 0.0:
            continue
        if abs(s - 1.0) > 1e-6:
            w1c /= s
            w2c /= s
            w3c /= s
            w4c /= s

        _emit({"type": "w_start", "w1": w1c, "w2": w2c, "w3": w3c, "w4": w4c})

        accs: List[float] = []
        for f in range(len(folds)):
            tr_idx, va_idx = fold_train_val_indices(folds, f)

            K1 = _cached_fold_grams(
                grams_cache,
                fold=f,
                spec=spec1,
                tr_idx=tr_idx,
                va_idx=va_idx,
                eps=float(normalisation_eps),
            )
            K2 = _cached_fold_grams(
                grams_cache,
                fold=f,
                spec=spec2,
                tr_idx=tr_idx,
                va_idx=va_idx,
                eps=float(normalisation_eps),
            )
            K3 = _cached_fold_grams(
                grams_cache,
                fold=f,
                spec=spec3,
                tr_idx=tr_idx,
                va_idx=va_idx,
                eps=float(normalisation_eps),
            )
            K4 = _cached_fold_grams(
                grams_cache,
                fold=f,
                spec=spec4,
                tr_idx=tr_idx,
                va_idx=va_idx,
                eps=float(normalisation_eps),
            )

            Ktr = (w1c * K1.K_tr) + (w2c * K2.K_tr) + (w3c * K3.K_tr) + (w4c * K4.K_tr)
            Kva_tr = (w1c * K1.K_va_tr) + (w2c * K2.K_va_tr) + (w3c * K3.K_va_tr) + (w4c * K4.K_va_tr)
            Ktr = np.asarray(Ktr, dtype=np.float32)
            Kva_tr = np.asarray(Kva_tr, dtype=np.float32)

            y_tr, y_va = y[tr_idx], y[va_idx]

            if model_key == "svm":
                clf = KernelSVM(
                    n_classes=int(n_classes),
                    kernel_fn=None,
                    lam=float(lam),
                    lr=float(lr),
                    epochs=int(epochs),
                    patience=int(patience),
                ).fit(
                    None,
                    y_tr,
                    K=Ktr,
                    y_val=y_va,
                    K_val=Kva_tr,
                )
                pred_va, _ = clf.predict(K_star=Kva_tr)
            else:
                clf = KernelRidgeRegression(
                    n_classes=int(n_classes),
                    kernel_fn=None,
                    lam=float(lam),
                    solver=str(krr_solver),
                    cg_tol=float(krr_cg_tol),
                ).fit(
                    None,
                    y_tr,
                    K=Ktr,
                )
                pred_va, _ = clf.predict(K_star=Kva_tr)

            accs.append(float(accuracy(y_va, pred_va)))

            _emit(
                {
                    "type": "fold_end",
                    "w1": w1c,
                    "w2": w2c,
                    "w3": w3c,
                    "w4": w4c,
                    "fold": int(f),
                    "n_folds": int(len(folds)),
                    "acc": float(accs[-1]),
                    "mean_so_far": float(np.mean(accs)),
                }
            )

        key = (float(w1c), float(w2c), float(w3c), float(w4c))
        fold_accs[key] = accs
        mean_accs[key] = float(np.mean(accs))

        _emit(
            {
                "type": "w_end",
                "w1": w1c,
                "w2": w2c,
                "w3": w3c,
                "w4": w4c,
                "mean_acc": float(mean_accs[key]),
                "fold_accs": accs,
            }
        )

    if len(mean_accs) == 0:
        raise ValueError(
            "No valid (w1,w2,w3) triples in w123_grid (must satisfy w1>=0, w2>=0, w3>=0, w1+w2+w3<=1)"
        )

    best_w = max(mean_accs, key=lambda ww: mean_accs[ww])
    _emit({"type": "search_end", "best_w": best_w, "best_mean_acc": float(mean_accs[best_w])})

    return {
        "best_w": best_w,
        "best_mean_acc": float(mean_accs[best_w]),
        "mean_accs": mean_accs,
        "fold_accs": fold_accs,
    }
