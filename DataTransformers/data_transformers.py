from sklearn.base import BaseEstimator, TransformerMixin
from skbio.stats.composition import clr, multi_replace

class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered Log-Ratio (CLR) Transformer with optional pseudo-count handling."""

    def __init__(self, pseudo_count=None):
        self.pseudo_count = pseudo_count
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.pseudo_count is not None:
            X = X + self.pseudo_count
        else:
            X = multi_replace(X)
            print("Multi")
        return clr(X)
    
    def __setstate__(self, state):
        if 'pseudo_count' not in state:
            state['pseudo_count'] = None
        self.__dict__.update(state)

class TSSTransformer(BaseEstimator, TransformerMixin):
    """Total Sum Scaling (TSS) Transformer."""

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.div(X.sum(axis=1), axis=0)
