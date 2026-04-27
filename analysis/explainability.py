import shap
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.utils.validation import check_is_fitted 

from logger import logger


class SHAPExplainer:    
    """
    Handles SHAP (SHapley Additive exPlanations) for model explainability.
    """

    def __init__(self, plotter, model, background_data=None, shap_explainer_params: dict = None):
        """
        Dynamically initializes the SHAP explainer based on the model class.

        Parameters
        ----------
        model : object
            Fitted model object.
        background_data : pd.DataFrame or np.ndarray, optional
            Background dataset needed by KernelExplainer and (optional) LinearExplainer.
        shap_explainer_params : dict, optional
            Additional params to be passed to the SHAP explainer constructor.
        """
        
        self.model = model
        self.logger = logger
        self.plotter = plotter

        if shap_explainer_params is None:
            shap_explainer_params = {}

        try:
            check_is_fitted(model)

        except NotFittedError:
            self.logger.error(f"Model {type(model)} was not fitted. Cannot compute SHAP values.")
            raise NotFittedError()

        if isinstance(model, Pipeline):
            model = model.named_steps['classifier']

        if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier, XGBClassifier)):
            self.explainer = shap.TreeExplainer(model, **shap_explainer_params)
            self.logger.info("Initialized TreeExplainer")

        elif isinstance(model, (LogisticRegression, RidgeClassifier)): 
            if background_data is not None:
                self.explainer = shap.LinearExplainer(model, masker = background_data, **shap_explainer_params)
                self.logger.info("Initialized LinearExplainer with background data")
            
            else:
                self.explainer = shap.LinearExplainer(model, **shap_explainer_params)
                self.logger.info("Initialized LinearExplainer")
        
        elif isinstance(model, (SVC, MLPClassifier)): 
            if background_data is None:
                self.logger.error("KernelExplainer needs 'background_data' to compute SHAP values")
                raise ValueError()

            if isinstance(model, SVC) and model.kernel == 'linear':
                # Use Linear Explainer if SVC kernel is linear, mathematically optimal
                self.explainer = shap.LinearExplainer(model, background_data, **shap_explainer_params) # type: ignore
                self.logger.info("Initialized LinearExplainer for SVC with linear kernel")
            
            else:
                self.explainer = shap.KernelExplainer(model, background_data, **shap_explainer_params) # type: ignore
                self.logger.info(f"Initialized KernelExplainer for {type(model).__name__}")

        else:
            self.logger.error(f"Model {type(model)} not supported for SHAPExplainer.")
            raise ValueError()

    
    def _extract(self, shap_values, index, negate):
        """
        Extracts the relevant SHAP values for binary classification.
        """

        if isinstance(shap_values, list) and index is not None:
            sv_rf = shap_values[index]

        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3 and index is not None:
            sv_rf = shap_values[:, :, index]
        else:
            sv_rf = shap_values

        return -sv_rf if negate else sv_rf
    

    def compute_shap_values(self, 
                            data: pd.DataFrame | np.ndarray, 
                            extract_index: int | list[int] = None, 
                            negate: bool = False,
                            visualize: bool = True) -> np.ndarray | list[np.ndarray]:
        """
        Computes SHAP values for the given data.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Data for which to compute SHAP values.
        extract_index : int or list[int], optional
            For binary classification, index of the class to extract SHAP values for. 
            If list, extracts for each index in the list. If None, returns all SHAP values.
        negate : bool, optional
            If True, negates the extracted SHAP values (useful for binary classification).7
        visualize : bool, optional
            Whether to display the plots.

        Returns
        -------
        shap_values : np.ndarray or list[np.ndarray]
            SHAP values for the given data. If extract_index is None, returns all SHAP values.
        """

        self.logger.info(f"Computing {self.model.__class__.__name__} SHAP values")
        
        shap_values = self.explainer.shap_values(data)

        if extract_index is None:
            self.logger.warning("No index specified, returning class 0 SHAP values")
            shap_values = self._extract(shap_values, 0, negate)

        elif isinstance(extract_index, int):
            shap_values = self._extract(shap_values, extract_index, negate)
        
        elif isinstance(extract_index, list):
            shap_values = [self._extract(sv, extract_index, negate) for sv in shap_values]
        
        else:
            self.logger.error(f"extract_index must be either None, int or list[int], got {type(extract_index)}")
            raise ValueError()
        
        self.plotter._shap_barplot_beeswarm(shap_values, data, visualize)

        return shap_values
    

    def get_top_features(self, shap_values: np.ndarray, index: pd.Index, head: int = 50):
        """
        Aggregates SHAP values across samples to identify the most important features.
        
        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values for the given data.
        index : pd.Index
            Feature names.
        head : int, optional
            Number of top features to return, default 50.

        Returns
        -------
        top_features : list
            List of top feature names.
        """

        mean_shap = np.abs(shap_values).mean(axis = 0)
        shap_importance = pd.Series(mean_shap, index = index)
        top = shap_importance.sort_values(ascending = False).head(head)

        return top.index.tolist()
