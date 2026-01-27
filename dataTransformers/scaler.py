from sklearn.preprocessing import StandardScaler

from dataTransformers.data_transformers import CLRTransformer
from logger.logger import logger

class Scaler:
    def __init__(self, transformer=CLRTransformer, scaler=StandardScaler):
        self.logger = logger
        self.transformer = transformer
        self.logger.info(f"Scaler: Using transformation {self.transformer.__name__}")
        self.scaler = scaler
        self.logger.info(f"Scaler: Using scaler {self.scaler.__name__}")
    
    def fit_transform(self, X):
        """Scales the data using the specified transformer and scaler"""
        X_clr = self.transformer().fit_transform(X)
        X_scaled = self.scaler().fit_transform(X_clr)
        return X_scaled