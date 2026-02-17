from sklearn.base import BaseEstimator, TransformerMixin
from skbio.stats.composition import clr, multi_replace


class CLRTransformer(BaseEstimator, TransformerMixin):
    """
    Centered Log-Ratio (CLR) Transformer with optional pseudo-count handling.
    
    Parameters
    ----------
        pseudo_count (int, default=None): Pseudo count

    """

    def __init__(self, pseudo_count=None):
        self.pseudo_count = pseudo_count
    
    def transform(self, data):
        """
        Applies CLR transformation to the input data.
        
        Parameters
        ----------
            data (pd.DataFrame): Input data to be transformed.
        
        Returns
        -------
            pd.DataFrame: Transformed data.
        """
        data = data + self.pseudo_count if self.pseudo_count is not None else multi_replace(data)
        return clr(data)
    
    def fit(self, x, y=None):
        return self

    def __setstate__(self, state):
        if 'pseudo_count' not in state:
            state['pseudo_count'] = None
        self.__dict__.update(state)


class TSSTransformer(BaseEstimator, TransformerMixin):
    """Total Sum Scaling (TSS) Transformer."""
    
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        """
        Applies TSS transformation to the input data.
        
        Parameters
        ----------
            data (pd.DataFrame): Input data to be transformed.
        
        Returns
        -------
            pd.DataFrame: Transformed data.
        """
        return data.div(data.sum(axis=1), axis=0)
