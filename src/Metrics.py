"""
Inputs (preferred names)
------------------------
labels : 1-D array-like of {0,1}
    1 = toxic. 
    0 = not toxix.

scores : 1-D array-like of float
    The model’s confidence for the *toxic* class for each sample.
    Larger values = “more toxic”. 
    Use probabilities or raw logits/margins.
    Do **not** pass hard 0/1 predictions.
    

subgroup_mask : 1-D array-like of bool
    True if the sample belongs to the identity subgroup, False otherwise.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------- utilities ---------------------------- #

def _safe_auc(labels_subset, scores_subset) :
    """
    Safe ROC-AUC: return np.nan instead of raising when a slice has only one class.
    """
    y = np.asarray(labels_subset, dtype=int)
    s = np.asarray(scores_subset, dtype=float)
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return np.nan


def _to_arrays(labels, scores, subgroup_mask):
    """Cast and align types once."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    g = np.asarray(subgroup_mask, dtype=bool)
    if not (y.ndim == s.ndim == g.ndim == 1) or not (len(y) == len(s) == len(g)):
        raise ValueError("labels, scores, subgroup_mask must be 1-D and of equal length")
    return y, s, g


# ---------------------------- metrics ---------------------------- #

def subgroup_auc(labels, scores, subgroup_mask) :
    """
    Subgroup AUC = AUC( D^-_g ∪ D^+_g )
    Measures separability *within the subgroup* itself.
    """
    y, s, g = _to_arrays(labels, scores, subgroup_mask)
    return _safe_auc(y[g], s[g])


def bpsn_auc(labels, scores, subgroup_mask) :
    """
    BPSN AUC = AUC( D^+ ∪ D^-_g )   (Background Positive, Subgroup Negative)
    Low values => subgroup negatives often outrank background positives (FP risk on subgroup).
    """
    y, s, g = _to_arrays(labels, scores, subgroup_mask)
    idx = (y == 1) | ((y == 0) & g)
    return _safe_auc(y[idx], s[idx])


def bnsp_auc(labels, scores, subgroup_mask) :
    """
    BNSP AUC = AUC( D^- ∪ D^+_g )   (Background Negative, Subgroup Positive)
    Low values => subgroup positives often fall below background negatives (FN risk on subgroup).
    """
    y, s, g = _to_arrays(labels, scores, subgroup_mask)
    idx = (y == 0) | ((y == 1) & g)
    return _safe_auc(y[idx], s[idx])


def positive_aeg(labels, scores, subgroup_mask) :
    """
    Positive AEG = AUC( S^+ vs B^+ ) - 0.5
    Threshold-free shift among positives (centered at 0).
      > 0: subgroup positives get higher scores than background positives.
      < 0: subgroup positives get lower scores.
    """
    y, s, g = _to_arrays(labels, scores, subgroup_mask)
    scores_subgroup_pos   = s[(y == 1) & g]
    scores_background_pos = s[(y == 1) & (~g)]
    if scores_subgroup_pos.size == 0 or scores_background_pos.size == 0:
        return np.nan

    tmp_labels = np.concatenate([
        np.ones(scores_subgroup_pos.size, dtype=int),
        np.zeros(scores_background_pos.size, dtype=int),
    ])
    tmp_scores = np.concatenate([scores_subgroup_pos, scores_background_pos])
    return 0.5 - _safe_auc(tmp_labels, tmp_scores)


def negative_aeg(labels, scores, subgroup_mask) :
    """
    Negative AEG = 0.5- AUC( S^- vs B^- )
    Threshold-free shift among negatives (centered at 0).
      > 0: subgroup negatives get higher scores (FP risk).
      < 0: subgroup negatives get lower scores.
    """
    y, s, g = _to_arrays(labels, scores, subgroup_mask)
    scores_subgroup_neg   = s[(y == 0) & g]
    scores_background_neg = s[(y == 0) & (~g)]
    if scores_subgroup_neg.size == 0 or scores_background_neg.size == 0:
        return np.nan

    tmp_labels = np.concatenate([
        np.ones(scores_subgroup_neg.size, dtype=int),
        np.zeros(scores_background_neg.size, dtype=int),
    ])
    tmp_scores = np.concatenate([scores_subgroup_neg, scores_background_neg])
    return 0.5 - _safe_auc(tmp_labels, tmp_scores) 

