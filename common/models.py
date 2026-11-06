"""
models.py
=========
Single definition of the classical ML classifiers and of their hyperparameter
grids, used by both pipelines:

  - IML/run_iml.py              (classifiers trained on the CLR-transformed taxa)
  - Orthrus/train_orthrus.py    (classifiers trained on the Orthrus latent space)

Hyperparameter optimization strategy (identical in both pipelines):
  GridSearchCV(model, grid, cv=INNER_CV, scoring=SCORING, n_jobs=-1)
  with cv=5 -> StratifiedKFold(n_splits=5, shuffle=False), refit on the whole
  outer training set.
"""

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

INNER_CV = 5            # GridSearchCV(cv=5) == StratifiedKFold(n_splits=5, shuffle=False) for classifiers
SCORING  = "accuracy"
N_JOBS   = -1

# Short names used on the command line / in output folders -> names used in the code
SHORT_NAMES = {"SVM": "Linear SVM", "LR": "Logistic Regression", "ET": "ExtraTrees"}


def get_ml_models_and_grids(random_state):
    """Models and hyperparameter grids (random_state = 42 + fold index)."""
    return {
        "Linear SVM": {
            "model":  LinearSVC(dual=False, max_iter=10000, random_state=random_state),
            "params": {
                'C':            [0.001, 0.01, 0.1, 1, 10, 100],
                'class_weight': [None, 'balanced']
            }
        },
        "Logistic Regression": {
            "model":  LogisticRegression(solver='liblinear', max_iter=10000, random_state=random_state),
            "params": {
                'C':            [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty':      ['l1', 'l2'],
                'class_weight': [None, 'balanced']
            }
        },
        "ExtraTrees": {
            "model":  ExtraTreesClassifier(random_state=random_state),
            "params": {
                'n_estimators':      [50, 100, 200, 500],
                'max_depth':         [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight':      [None, 'balanced']
            }
        }
    }


def inner_cv_splitter():
    """The splitter that GridSearchCV(cv=INNER_CV) uses for a classifier."""
    return StratifiedKFold(n_splits=INNER_CV)
