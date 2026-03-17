"""
start.py — Reproduce the HOG+HSV+CM+LBP submission.

Trains on the full training set using pre-tuned kernel weights
(w_hog=0.30, w_hsv=0.15, w_cm=0.20, w_lbp=0.35) and saves
predictions to Yte.csv in the project root.
"""

import numpy as np
import pandas as pd
from functools import partial

from src.data import encode_labels, load_data
from src.features import HOG, HSVHistogram, ColorMoments, UniformLBPHistogram
from src.kernel_normalisation import unit_diag
from src.kernels import (
    chi2_rbf_kernel_matrix_channel,
    estimate_chi2_gammas_channel,
    gaussian_kernel_matrix,
    estimate_gamma,
    laplacian_kernel_matrix,
    estimate_laplacian_gamma,
)
from src.kernel_normalisation import normalise_train_gram, normalise_cross_gram
from src.krr import KernelRidgeRegression

# ── Pre-tuned weights ──────────────────────────────────────────────────────────
W_HOG = 0.30
W_HSV = 0.15
W_CM  = 0.20
W_LBP = 0.35

DATA_DIR = "data/"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
X_tr, X_te, y_tr_raw = load_data(DATA_DIR)
y_tr, inv_map = encode_labels(y_tr_raw)
n_classes = len(np.unique(y_tr))

# ── Extract features ───────────────────────────────────────────────────────────
print("Extracting HOG features...")
hog = HOG(
    image_shape=(3, 32, 32),
    colour_mode="rgb",
    cell_size=8,
    block_size=2,
    block_stride=1,
    num_bins=6,
    signed=False,
).fit(X_tr)
X_tr_hog = hog.transform(X_tr)
X_te_hog = hog.transform(X_te)

print("Extracting HSV Histogram features...")
hsv = HSVHistogram(bins=(8, 8, 8), grid=(4, 4)).fit(X_tr)
X_tr_hsv = hsv.transform(X_tr)
X_te_hsv = hsv.transform(X_te)

print("Extracting Color Moment features...")
cm = ColorMoments(grid=(8, 8)).fit(X_tr)
X_tr_cm = cm.transform(X_tr)
X_te_cm = cm.transform(X_te)

print("Extracting LBP features...")
lbp = UniformLBPHistogram(cell_size=8, pool="concat", per_channel=True).fit(X_tr)
X_tr_lbp = lbp.transform(X_tr)
X_te_lbp = lbp.transform(X_te)

# ── Build kernels on full training set ─────────────────────────────────────────
print("Computing kernels...")
gammas_hog = estimate_chi2_gammas_channel(X_tr_hog)
hog_chi2 = partial(chi2_rbf_kernel_matrix_channel, gammas=gammas_hog)

gamma_hsv = estimate_gamma(X_tr_hsv)
hsv_gaussian = partial(gaussian_kernel_matrix, gamma=gamma_hsv)

gamma_cm = estimate_laplacian_gamma(X_tr_cm)
cm_laplacian = partial(laplacian_kernel_matrix, gamma=gamma_cm)

gamma_lbp = estimate_gamma(X_tr_lbp)
lbp_gaussian = partial(gaussian_kernel_matrix, gamma=gamma_lbp)

Kh_tr = normalise_train_gram(hog_chi2(X_tr_hog, X_tr_hog),   unit_diag(X_tr_hog))
Ks_tr = normalise_train_gram(hsv_gaussian(X_tr_hsv, X_tr_hsv), unit_diag(X_tr_hsv))
Kc_tr = normalise_train_gram(cm_laplacian(X_tr_cm, X_tr_cm),   unit_diag(X_tr_cm))
Kl_tr = normalise_train_gram(lbp_gaussian(X_tr_lbp, X_tr_lbp), unit_diag(X_tr_lbp))

Ktr = W_HOG * Kh_tr + W_HSV * Ks_tr + W_CM * Kc_tr + W_LBP * Kl_tr

# ── Train ──────────────────────────────────────────────────────────────────────
print("Training KRR model...")
model = KernelRidgeRegression(n_classes=n_classes, kernel_fn=None, lam=1e-4).fit(
    None, y_tr, K=Ktr
)

# ── Predict on test set ────────────────────────────────────────────────────────
print("Computing test kernel and predicting...")
Kh_te = normalise_cross_gram(hog_chi2(X_te_hog, X_tr_hog),     unit_diag(X_te_hog),  unit_diag(X_tr_hog))
Ks_te = normalise_cross_gram(hsv_gaussian(X_te_hsv, X_tr_hsv), unit_diag(X_te_hsv),  unit_diag(X_tr_hsv))
Kc_te = normalise_cross_gram(cm_laplacian(X_te_cm, X_tr_cm),   unit_diag(X_te_cm),   unit_diag(X_tr_cm))
Kl_te = normalise_cross_gram(lbp_gaussian(X_te_lbp, X_tr_lbp), unit_diag(X_te_lbp), unit_diag(X_tr_lbp))

K_star = W_HOG * Kh_te + W_HSV * Ks_te + W_CM * Kc_te + W_LBP * Kl_te

yte_int, _ = model.predict(K_star=K_star)
yte = np.array([inv_map[i] for i in yte_int])

# ── Save predictions ───────────────────────────────────────────────────────────
sub = pd.DataFrame({"Prediction": yte})
sub.index += 1
sub.to_csv("Yte.csv", index_label="Id")
print("Saved predictions to Yte.csv")
