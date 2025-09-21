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
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ---------------------------- utilities ---------------------------- #

def _safe_auc(labels_subset, scores_subset):
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

def subgroup_auc(labels, scores, subgroup_mask):
    """
    Subgroup AUC = AUC( D^-_g ∪ D^+_g )
    Measures separability *within the subgroup* itself.
    """
    y, s, g = _to_arrays(labels, scores, subgroup_mask)
    return _safe_auc(y[g], s[g])


def bpsn_auc(labels, scores, subgroup_mask):
    """
    BPSN AUC = AUC( D^+ ∪ D^-_g )   (Background Positive, Subgroup Negative)
    Low values => subgroup negatives often outrank background positives (FP risk on subgroup).
    """
    y, s, g = _to_arrays(labels, scores, subgroup_mask)
    idx = (y == 1) | ((y == 0) & g)
    return _safe_auc(y[idx], s[idx])


def bnsp_auc(labels, scores, subgroup_mask):
    """
    BNSP AUC = AUC( D^- ∪ D^+_g )   (Background Negative, Subgroup Positive)
    Low values => subgroup positives often fall below background negatives (FN risk on subgroup).
    """
    y, s, g = _to_arrays(labels, scores, subgroup_mask)
    idx = (y == 0) | ((y == 1) & g)
    return _safe_auc(y[idx], s[idx])


def positive_aeg(labels, scores, subgroup_mask):
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
    return _safe_auc(tmp_labels, tmp_scores) - 0.5


def negative_aeg(labels, scores, subgroup_mask):
    """
    Negative AEG = AUC( S^- vs B^- ) - 0.5
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
    return _safe_auc(tmp_labels, tmp_scores) - 0.5


# ========= tiny training demo ========= #
def main():
    rng = 0
    n = 800

    # 1) Toy dataset
    X, y = make_classification(
        n_samples=n, n_features=8, n_informative=5, n_redundant=1,
        class_sep=1.2, flip_y=0.02, weights=[0.6, 0.4], random_state=rng
    )

    # Create a toy binary "gender" attribute: True=female, False=male (~50/50)
    gender = (np.random.RandomState(rng).rand(n) < 0.5)

    # Optional: inject a small bias by shifting features differently per gender
    shift = np.zeros_like(X)
    shift[gender]  -= 0.25   # hurts females a bit (↓ scores ⇒ potential FN risk)
    shift[~gender] += 0.05   # slight boost for males
    X_biased = X + shift

    # Split, keeping gender aligned
    X_tr, X_te, y_tr, y_te, gender_tr, gender_te = train_test_split(  
        X_biased, y, gender, test_size=0.3, random_state=rng, stratify=y
    )

    # 2) Train a tiny model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)

    # 3) Scores for the toxic class (class 1)
    scores = clf.predict_proba(X_te)[:, 1]  # use probabilities, not hard 0/1 preds

    # 3.5) Gender subgroup masks , we should define a mask for each subgroup
    masks = {
        "female": gender_te.astype(bool),      # True = female
        "male":   (~gender_te).astype(bool),   # True = male
    }

    # 4) Print metrics
    print("\n=== Mini model fairness report (toy: gender) ===")
    print(f"Test size: {len(y_te)} | Pos rate: {y_te.mean():.3f}")

    for name, gmask in masks.items():
        if gmask.dtype != bool or len(gmask) != len(y_te):
            raise ValueError(f"Mask '{name}' must be boolean and same length as y_te/scores")

        print(f"\n-- {name.upper()} --")
        print(f"Subgroup size: {int(gmask.sum())} ({gmask.mean():.3f} of test)")
        print("subgroup_auc :", subgroup_auc(y_te, scores, gmask))
        print("bpsn_auc     :", bpsn_auc(y_te, scores, gmask))
        print("bnsp_auc     :", bnsp_auc(y_te, scores, gmask))
        print("positive_aeg :", positive_aeg(y_te, scores, gmask))
        print("negative_aeg :", negative_aeg(y_te, scores, gmask))


if __name__ == "__main__":
    main()
