"""
preprocessing.py
================
Data loading and preprocessing shared by the IML pipeline and by Orthrus.

The dataset (datasets/raw_dataset.csv) contains one row per patient: metadata
columns (samples, BioProject, ORR, response, ...) and 4,630 taxa columns
(names starting with 'k__'), with relative abundances already thresholded
(values < 10 set to zero) and TSS-normalized to a total of 100.

Both pipelines then apply:
  1. a CLR transformation, which acts on each sample independently with fixed
     constants (no parameter is estimated from other samples);
  2. standard scaling, the only data-dependent step, always fitted on the
     training data only.

In both pipelines the zeros are handled by a fixed pseudocount of 1e-6
(clr_pseudocount). In the IML pipeline CLR and scaling are wrapped in a
scikit-learn transformer (CLRScaler) that is part of the tuned pipeline, so the
scaler is refit in every inner fold; in Orthrus the StandardScaler is fitted
on the outer training set.
"""

import csv

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
TAXA_PREFIX  = "k__"
LABEL_COL    = "response"
PSEUDO_COUNT = 1e-6


# ==========================================
# Data loading
# ==========================================
def _infer_separator(path):
    """Comma or semicolon, inferred from the first 4 KB of the file."""
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return csv.Sniffer().sniff(f.read(4096), delimiters=",;").delimiter
    except csv.Error:
        return ","


def load_dataset(path):
    """
    Reads the dataset and removes the patients without response label.

    Returns
    -------
    df        : DataFrame in the original row order (index reset)
    taxa_cols : list of the taxa columns (names starting with 'k__')
    """
    df = pd.read_csv(path, sep=_infer_separator(path), decimal=",", low_memory=False)
    df = df.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    taxa_cols = [c for c in df.columns if c.startswith(TAXA_PREFIX)]
    df[taxa_cols] = df[taxa_cols].apply(pd.to_numeric, errors="coerce")
    return df, taxa_cols


# ==========================================
# CLR with a fixed pseudocount (both pipelines)
# ==========================================
def clr_pseudocount(X_raw, pseudo_count=PSEUDO_COUNT):
    """CLR with a fixed pseudocount. Sample-wise, so it can be applied before the split."""
    log_X = np.log(X_raw + pseudo_count)
    return log_X - np.mean(log_X, axis=1, keepdims=True)


# ==========================================
# IML: CLR + standard scaling, as a scikit-learn transformer
# ==========================================
class CLRTransformer(BaseEstimator, TransformerMixin):
    """CLR transformation with a fixed pseudocount. Sample-wise, no fitting."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = clr_pseudocount(X.values)
        return pd.DataFrame(values, index=X.index, columns=X.columns)


class CLRScaler(BaseEstimator, TransformerMixin):
    """
    CLR transformation followed by StandardScaler (fitted in fit, i.e. on the
    training data only). Used as the first step of the IML pipeline.
    """

    def fit(self, X, y=None):
        self.clr_    = CLRTransformer()
        self.scaler_ = StandardScaler().fit(self.clr_.transform(X), y)
        return self

    def transform(self, X):
        scaled = self.scaler_.transform(self.clr_.transform(X))
        return pd.DataFrame(scaled, index=X.index, columns=X.columns)
