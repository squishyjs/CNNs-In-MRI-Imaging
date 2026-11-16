"""Multi-plane fusion: an L2-regularised logistic regression per
model, seed and task, fitted on tuning-set predictions and applied unchanged
to the test set. Falls back to the single available plane when a model was
trained on fewer than three planes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _plane_matrix(preds_by_plane: dict[str, pd.DataFrame], task: str):
    """Align per-plane prediction frames on exam id -> (ids, X, y)."""
    planes = sorted(preds_by_plane)
    base = preds_by_plane[planes[0]][["id", f"y_{task}", f"p_{task}"]].rename(
        columns={f"p_{task}": planes[0]})
    for p in planes[1:]:
        base = base.merge(preds_by_plane[p][["id", f"p_{task}"]].rename(
            columns={f"p_{task}": p}), on="id")
    return base["id"].values, base[planes].values, base[f"y_{task}"].values


def fuse_planes(tune_preds: dict[str, pd.DataFrame], test_preds: dict[str, pd.DataFrame],
                task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on tuning predictions, apply unchanged to both sets.
    Returns (fused_tuning, fused_test) frames with columns id, y, p."""
    ids_tu, X_tu, y_tu = _plane_matrix(tune_preds, task)
    ids_te, X_te, y_te = _plane_matrix(test_preds, task)
    if len(tune_preds) == 1:                      # single plane: identity fusion
        return (pd.DataFrame({"id": ids_tu, "y": y_tu, "p": X_tu[:, 0]}),
                pd.DataFrame({"id": ids_te, "y": y_te, "p": X_te[:, 0]}))
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X_tu, y_tu)
    return (pd.DataFrame({"id": ids_tu, "y": y_tu, "p": lr.predict_proba(X_tu)[:, 1]}),
            pd.DataFrame({"id": ids_te, "y": y_te, "p": lr.predict_proba(X_te)[:, 1]}))


def seed_average(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average fused probabilities across seeds to one prediction per exam (§3.8)."""
    merged = frames[0][["id", "y"]].copy()
    probs = pd.concat([f.set_index("id")["p"] for f in frames], axis=1)
    merged["p"] = merged["id"].map(probs.mean(axis=1))
    return merged
