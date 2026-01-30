from sklearn.preprocessing import StandardScaler

from dataTransformers.data_transformers import CLRTransformer
from logger.logger import logger

class Scaler:
    """
    Scales the data using the specified transformer and scaler.

    Parameters
    ----------
        transformer (class): The specific data transfomer to use (e.g., CLRTransformer).
        scaler (class): The specific scaler to use (e.g., StandardScaler).
    """

    def __init__(self, transformer=CLRTransformer, scaler=StandardScaler):
        self.logger = logger
        
        self.transformer = transformer
        self.logger.info(f"Scaler: Using transformation {self.transformer.__name__}")
        
        self.scaler = scaler
        self.logger.info(f"Scaler: Using scaler {self.scaler.__name__}")
    
    def fit_transform(self, data):
        """
        Scales the data using the specified transformer and scaler.
        
        Parameters
        ----------
            data (pd.DataFrame): Input data to scale
            
        Returns
        -------
            scaled_data (pd.DataFrame): Scaled data
        """
        clr_data = self.transformer().fit_transform(data)
        return self.scaler().fit_transform(clr_data)