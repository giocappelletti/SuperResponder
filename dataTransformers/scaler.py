from sklearn.preprocessing import StandardScaler
import pandas as pd

from dataTransformers.data_transformers import CLRTransformer
from logger.logger import logger


class SmartScaler:
    """
    Scales the data using the specified transformer and scaler.

    Parameters
    ----------
        transformer: class
            The specific data transfomer to use (e.g., CLRTransformer).
        scaler: class
            The specific scaler to use (e.g., StandardScaler).
    """

    def __init__(self, transformer=CLRTransformer, scaler=StandardScaler):
        self.logger = logger
        
        self.transformer = transformer()
        self.logger.info(f"Scaler: Using transformer {self.transformer!r}")
        
        self.scaler = scaler(with_mean=False, with_std=False)
        self.logger.info(f"Scaler: Using scaler {self.scaler!r}") 
    

    def fit_transform(self, data: pd.DataFrame):
        """
        Fits and transforms the data using the specified transformer and scaler.
        
        Parameters
        ----------
            data: pd.DataFrame
                Input data to scale and transform
            
        Returns
        -------
            pd.DataFrame: 
                Scaled and transformed data
        """
        transf_data = self.transformer.fit_transform(data)
        return self.scaler.fit_transform(transf_data)


    def transform(self, data):
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
        transf_data = self.transformer.transform(data)
        return self.scaler.transform(transf_data)
    

    def transform_then_fit_transform(self, data):
        """
        Transforms then fits the data using the specified transformer and scaler.
        
        Parameters
        ----------
            data: pd.DataFrame
                Input data to scale and transform
            
        Returns
        -------
            pd.DataFrame: 
                Scaled and transformed data
        """
        transf_data = self.transformer.transform(data)
        return self.scaler.fit_transform(transf_data)
