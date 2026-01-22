from DataTransformers.data_transformers import CLRTransformer

from sklearn.preprocessing import StandardScaler

class Scaler:
    def __init__(self, transformer=CLRTransformer, scaler=StandardScaler):
        self.transformer = transformer
        self.scaler = scaler
    
    def fit_transform(self, X):
        """Scales the data using the specified transformer and scaler"""
        X_clr = self.transformer().fit_transform(X)
        X_scaled = self.scaler().fit_transform(X_clr)
        
        return X_scaled