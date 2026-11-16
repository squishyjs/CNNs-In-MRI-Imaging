"""Evaluation metrics and statistics: AUC with DeLong confidence
intervals, paired DeLong tests, Youden thresholds, sensitivity/specificity with
Wilson intervals, F1, Brier score, expected calibration error, BH correction.

DeLong implementation follows the fast algorithm of Sun & Xu (2014) as used by
the open-source fast-DeLong implementations referenced in implementation guide
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (brier_score_loss, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)


# fast DeLong

def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _delong(y_true: np.ndarray, scores: np.ndarray):
    """scores: (k, n) matrix of k classifiers. Returns (aucs, covariance matrix)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    order = np.argsort(-y_true)                 # positives first
    scores = scores[:, order]
    m = int(y_true.sum())
    n = len(y_true) - m
    k = scores.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(scores[r, :m])
        ty[r] = _midrank(scores[r, m:])
        tz[r] = _midrank(scores[r])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_ci(y_true, y_prob, alpha: float = 0.95):
    """AUC and DeLong 95% CI for one classifier."""
    aucs, cov = _delong(y_true, y_prob)
    se = np.sqrt(max(float(cov[0, 0]), 0.0))
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    lo, hi = np.clip([aucs[0] - z * se, aucs[0] + z * se], 0, 1)
    return float(aucs[0]), (float(lo), float(hi))


def delong_test(y_true, p1, p2):
    """Two-sided paired DeLong test comparing two correlated AUCs on the same exams."""
    aucs, cov = _delong(y_true, np.vstack([p1, p2]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0] - aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0] - aucs[1]), float(2 * stats.norm.sf(abs(z)))


# thresholded metrics

def youden_threshold(y_true, y_prob) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    thr = np.where(np.isinf(thr), 1.0, thr)
    return float(thr[np.argmax(tpr - fpr)])


def wilson_ci(k: int, n: int, alpha: float = 0.95):
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=1 - alpha, method="wilson")
    return float(lo), float(hi)


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins (§3.8)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(e)


def reliability_bins(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append(dict(bin=b, confidence=y_prob[m].mean(),
                             frequency=y_true[m].mean(), count=int(m.sum())))
    return pd.DataFrame(rows)


def full_metrics(y_true, y_prob, threshold: float) -> dict:
    """Everything Table 4.2 needs for one model and task."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_hat = (y_prob >= threshold).astype(int)
    auc, (lo, hi) = delong_ci(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return dict(
        auc=auc, auc_lo=lo, auc_hi=hi, threshold=threshold,
        sensitivity=sens, sens_ci=wilson_ci(tp, max(tp + fn, 1)),
        specificity=spec, spec_ci=wilson_ci(tn, max(tn + fp, 1)),
        f1=f1_score(y_true, y_hat, zero_division=0),
        brier=brier_score_loss(y_true, y_prob),
        ece=ece(y_true, y_prob),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def bh_correct(pvals: list[float], alpha: float = 0.05):
    """Benjamini-Hochberg FDR correction (statsmodels, method='fdr_bh')."""
    from statsmodels.stats.multitest import multipletests
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return reject, p_adj


def report_at_threshold(y_true, y_prob, threshold, target_names=("negative", "positive")):
    y_hat = (np.asarray(y_prob) >= threshold).astype(int)
    return classification_report(np.asarray(y_true).astype(int), y_hat,
                                 target_names=list(target_names), zero_division=0)
