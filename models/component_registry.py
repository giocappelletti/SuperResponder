from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA, KernelPCA

from preprocessing import CLRTransformer, TSSTransformer, CompReduction

# Mapping for transformation components
TRANSFORMATION_MAP = {
    "CLR": CLRTransformer,
    "TSS": TSSTransformer,
}

# Mapping for scaler components
SCALER_MAP = {
    "Standard": StandardScaler,
    "Robust": RobustScaler,
    "MinMax": MinMaxScaler,
}

# Mapping for classifier components
CLASSIFIER_MAP = {
    "LR": LogisticRegression,
    "Ridge": RidgeClassifier,
    "SVM": SVC,
    "MLP": MLPClassifier,
    "XGB": XGBClassifier,
    "RF": RandomForestClassifier,
    "ET": ExtraTreesClassifier,
}

# Mapping for reduction components
REDUCTION_MAP = {
    "Comp": CompReduction,
    "PCA": PCA,
    "KPCA": KernelPCA,
}