"""
shap_explainer.py
=================
SHAP values of the IML classifiers:
  - LinearExplainer for Logistic Regression and Linear SVM, with the training
    data of the model as background (masker);
  - TreeExplainer for Extra Trees.
Both give exact SHAP values for the corresponding model class.
"""

import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def shap_values_positive_class(model, X_background, X_explain):
    """
    SHAP values of `model` on `X_explain` for the positive class (Responder).

    model        : fitted LogisticRegression, LinearSVC or ExtraTreesClassifier
    X_background : training data of the model (background of the LinearExplainer)
    X_explain    : data to explain
    """
    if isinstance(model, ExtraTreesClassifier):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, (LogisticRegression, LinearSVC)):
        explainer = shap.LinearExplainer(model, masker=X_background)
    else:
        raise ValueError(f"Model {type(model).__name__} not supported")

    values = explainer.shap_values(X_explain)
    # binary classifiers: keep the values of class 1 when one array per class is returned
    if isinstance(values, list):
        return values[1]
    if values.ndim == 3:
        return values[:, :, 1]
    return values
