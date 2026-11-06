"""
splits.py
=========
Single definition of the outer train/test splits, used by both the IML pipeline
and Orthrus. It depends only on the dataset, so the two pipelines obtain exactly
the same test samples for every fold, also when run on different machines.

Modes:
  - "holdout": N stratified 80/20 splits (N = 5 in the paper),
               fold i = train_test_split(random_state=42+i, stratify=response).
  - "loco":    Leave-One-Cohort-Out over the BioProject column (9 folds).

Splits are returned as sample IDs (column 'samples'), so they do not depend on
the row order with which each script loads the dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ID_COL     = "samples"
LABEL_COL  = "response"
COHORT_COL = "BioProject"

HOLDOUT_TEST_SIZE    = 0.2
HOLDOUT_RANDOM_STATE = 42


def _encode_labels(df):
    # LabelEncoder sorts the classes: 'NR' -> 0, 'R' -> 1
    return LabelEncoder().fit_transform(df[LABEL_COL])


def get_splits(df: pd.DataFrame, mode: str = "holdout", n_splits: int = 5):
    """
    Parameters
    ----------
    df : DataFrame returned by preprocessing.load_dataset (original row order,
         rows with missing 'response' removed).
    mode : "holdout" or "loco".
    n_splits : number of splits for "holdout" (ignored for "loco").

    Returns
    -------
    list of dicts: {"name": str, "train_ids": np.ndarray, "test_ids": np.ndarray}
    """
    ids = df[ID_COL].to_numpy()

    if mode == "holdout":
        y = _encode_labels(df)
        idx = np.arange(len(df))
        splits = []
        for i in range(n_splits):
            tr, te = train_test_split(idx, test_size=HOLDOUT_TEST_SIZE,
                                      random_state=HOLDOUT_RANDOM_STATE + i, stratify=y)
            splits.append({"name": f"fold_{i + 1}", "train_ids": ids[tr], "test_ids": ids[te]})
        return splits

    if mode == "loco":
        cohorts = df[COHORT_COL].to_numpy()
        splits = []
        for c in sorted(np.unique(cohorts)):
            test_mask = cohorts == c
            splits.append({"name": f"loco_{c}",
                           "train_ids": ids[~test_mask],
                           "test_ids":  ids[test_mask]})
        return splits

    raise ValueError(f"Unknown mode: {mode!r} (use 'holdout' or 'loco')")


def positions(df: pd.DataFrame, sample_ids) -> np.ndarray:
    """Converts sample IDs into positional indices of df (in the order of the IDs)."""
    pos = pd.Index(df[ID_COL]).get_indexer(sample_ids)
    if (pos < 0).any():
        raise KeyError("Some sample IDs of the split are not present in the DataFrame")
    return pos


def save_test_ids(out_dir: str, split: dict):
    """Saves the test sample IDs of the fold (test_samples.csv), so that the folds of the
    two pipelines can be checked to be identical."""
    os.makedirs(out_dir, exist_ok=True)
    pd.Series(np.sort(split["test_ids"]), name=ID_COL).to_csv(
        os.path.join(out_dir, "test_samples.csv"), index=False)
