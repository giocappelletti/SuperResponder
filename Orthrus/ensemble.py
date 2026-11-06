"""
ensemble.py
===========
Majority-vote Ensemble used by Orthrus/train_orthrus.py.

Voters: the three classical classifiers (Linear SVM, LR, ET) trained on the
Orthrus latent space and the Orthrus classification head. With 4 voters a 2-2
tie is possible: in that case the prediction of the Orthrus head is used.
"""

import numpy as np


def majority_vote(voters, tiebreak=None):
    """
    voters   : list of 1-D arrays of 0/1 predictions (same length)
    tiebreak : 1-D array of 0/1 predictions used where the vote is tied
               (the Orthrus head); if None, ties go to class 0
    """
    votes = np.vstack(voters).astype(int)
    n_pos = votes.sum(axis=0)
    n_voters = votes.shape[0]
    pred = (2 * n_pos > n_voters).astype(int)          # strict majority for class 1
    tie = (2 * n_pos == n_voters)
    if tiebreak is not None:
        pred[tie] = np.asarray(tiebreak).astype(int)[tie]
    return pred
