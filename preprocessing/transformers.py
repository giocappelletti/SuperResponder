import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from skbio.stats.composition import clr, multi_replace
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from logger import logger


class CLRTransformer(BaseEstimator, TransformerMixin):
    """
    Centered Log-Ratio (CLR) Transformer with optional pseudo-count handling.
    
    Parameters
    ----------
        pseudo_count: int, default=None
            Pseudo count to be added to data

    """

    def __init__(self, pseudo_count = None):
        self.pseudo_count = pseudo_count
    

    def transform(self, data):
        """
        Applies CLR transformation to the input data.
        """
        data = data + self.pseudo_count if self.pseudo_count is not None else multi_replace(data)
        return clr(data)
    

    def fit(self, x, y = None): # Must write x and y to avoid errors
        """
        Fits the transformer. No-op for CLRTransformer as it doesn't learn parameters from data.
        """
        return self


    def __setstate__(self, state):
        if 'pseudo_count' not in state:
            state['pseudo_count'] = None
        self.__dict__.update(state)


class TSSTransformer(BaseEstimator, TransformerMixin):
    """Total Sum Scaling (TSS) Transformer."""
    
    def fit(self, x, y = None):
        """
        Fits the transformer. No-op for TSSTransformer as it doesn't learn parameters from data.
        """
        return self


    def transform(self, data):
        """
        Applies TSS transformation to the input data.
        """
        return data.div(data.sum(axis = 1), axis = 0)


class PLSVarianceSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector and dimensionality reducer based on Partial Least Squares (PLS) Regression.
    It selects the number of components required to explain a specified threshold of variance.

    Parameters
    ----------
        variance_threshold: float, default=0.9
            The cumulative variance threshold to determine the number of components.
        n_components: int, default=None
            If specified, overrides the variance_threshold and uses a fixed number of components.
    """

    def __init__(self, variance_threshold = 0.9, n_components = None):
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self._pls = None


    def fit(self, X, y):
        """
        Fits the PLS model and determines the optimal number of components based on variance threshold.
        """
        
        components = X.shape[1] if self.n_components == None else self.n_components

        self._pls = PLSRegression(n_components = components)
        self._pls.fit(X, y)

        # Compute variance        
        X_transformed = self._pls.transform(X)

        if self.n_components == None:
            explained_variance = np.var(X_transformed, axis = 0)
            total_var = explained_variance.sum()
            explained_ratio = explained_variance / total_var
            cumulative_variance = np.cumsum(explained_ratio) 

            if np.any(cumulative_variance >= self.variance_threshold):
                self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            else:
                self.n_components = components

        return self


    def transform(self, X):
        """
        Transforms the input data using the fitted PLS model.
        """
        X_transformed = self._pls.transform(X)
        return X_transformed[:, :self.n_components]


class CompReduction(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction wrapper for compositional features.

    Applies a specified reduction technique only to the last `n_comp_features` columns 
    of the input data, preserving the initial columns (metadata).

    Parameters
    ----------
        n_comp_features: int
            The number of compositional features located at the end of the feature set.
        reduction: estimator
            The dimensionality reduction instance (e.g., PCA, PLSVarianceSelector) to apply.
            
    """

    def __init__(self, n_comp_features = 1128, reduction = PCA()):
        self.n_comp_features = n_comp_features
        self.reduction = reduction


    def fit(self, X, y = None):
        """
        Fits the reduction technique on the compositional features.
        """

        X_comp = X[:, -self.n_comp_features:]
        self.reduction.fit(X_comp)
        
        return self


    def transform(self, X):
        """
        Applies the reduction technique to the compositional features and concatenates them with metadata.
        """

        X_meta = X[:, :-self.n_comp_features]
        X_comp = X[:, -self.n_comp_features:]
        X_comp_reduced = self.reduction.transform(X_comp)
        
        return np.hstack((X_meta, X_comp_reduced))


class SmartScaler(BaseEstimator, TransformerMixin):
    """
    Scales the data using the specified transformer and scaler.

    Parameters
    ----------
        transformer: class
            The specific data transfomer to use (e.g., CLRTransformer).
        scaler: class
            The specific scaler to use (e.g., StandardScaler).
        with_mean: bool, default=False
            Whether to center the data before scaling.
        with_std: bool, default=False
            Whether to scale the data to unit variance.
    """

    def __init__(self, transformer=CLRTransformer, scaler=StandardScaler, with_mean=True, with_std=True):
        self.logger = logger
        
        self.transformer = transformer
        self.scaler = scaler           
        self.with_mean = with_mean     
        self.with_std = with_std       

        self._transformer_instance = None
        self._scaler_instance = None


    def _instantiate_components(self):
        """Instantiate transformer and scaler if not already done."""
        
        if self._transformer_instance is None:
            # Instantiate the transformer class
            self._transformer_instance = self.transformer()

        if self._scaler_instance is None:
            # Pass with_mean/with_std only if the scaler supports them
            scaler_kwargs = {}
            # Check if the scaler's __init__ method accepts 'with_mean' and 'with_std'
            if 'with_mean' in self.scaler.__init__.__code__.co_varnames:
                scaler_kwargs['with_mean'] = self.with_mean
            if 'with_std' in self.scaler.__init__.__code__.co_varnames:
                scaler_kwargs['with_std'] = self.with_std
            # Instantiate the scaler class with appropriate kwargs
            self._scaler_instance = self.scaler(**scaler_kwargs)
    

    def fit(self, X, y=None):
        """
        Fits the transformer and scaler on the provided data.
        """
        self._instantiate_components() # Ensure components are instantiated
        transf_data = self._transformer_instance.fit_transform(X, y) # Use instance and fit_transform
        self._scaler_instance.fit(transf_data, y) # Use instance and pass y
        return self
    

    def transform(self, data) -> pd.DataFrame:
        """
        Transforms the data using the specified transformer and scaler.
        
        Parameters
        ----------
            data: pd.DataFrame
                Input data to transform
            
        Returns
        -------
            pd.DataFrame: 
                Transformed data
        """
        # Store original column names
        original_columns = data.columns

        self._instantiate_components() # Ensure components are instantiated
        transf_data = self._transformer_instance.transform(data) # Use instance
        scaled_array = self._scaler_instance.transform(transf_data) # Use instance
        return pd.DataFrame(scaled_array, index=data.index, columns=original_columns)