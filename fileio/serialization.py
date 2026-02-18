import os
import time
from typing import Literal
import pandas as pd

from logger.logger import logger


class Serialization:
    """
    Serialize test results to files.
    
    Parameters
    ----------
        base_dir: str
            Directory to store results
        cache_dir: str
            Directory to store cached datasets in feather format.
    """

    def __init__(self, base_dir="./results", cache_dir=".cached_datasets"):
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.logger = logger

    def cache_dataset(self, dataset: pd.DataFrame, path: str):
        """
        Save optimized feather file for next usage.
        """
        self.logger.info(f"Caching dataset to {path} for future use")
        os.makedirs(os.path.join(self.cache_dir, os.path.dirname(path)), exist_ok=True)
        
        try:
            dataset = dataset.copy()
            dataset.to_feather(os.path.join(self.cache_dir, f"{path}.feather"))
            
            self.logger.info(f"Dataset cached successfully")

        except Exception as e:
            self.logger.warning(f"Failed to cache dataset to {path}: {e}")

    def write_to_disk(self, 
                      data: pd.DataFrame, 
                      path: str, 
                      save_format: Literal['csv', 'tsv', 'xlsx'],
                      index: bool = True):
        """
        Write results to file.
        """
        
        self.logger.info(f"Saving results to file: {path}.{save_format}")

        try:            
            if save_format in ["csv", "tsv"]:
                separator = "," if save_format == "csv" else "\t"
                data.to_csv(f"{path}.{save_format}", sep=separator, index=index)
            
            elif save_format == "xlsx":
                data.to_excel(f"{path}.{save_format}", index=index)
        
        except Exception as e:                                                                                          
            self.logger.error(f"Error saving file: {e}")


    def save_file(self, 
                  data: pd.DataFrame, 
                  subfolder: str, 
                  exp_type: str, 
                  exp_group: str, 
                  save_format: Literal['csv', 'tsv', 'xlsx'],
                  distance_type: str = None,
                  should_save_time=True) -> str | None:
        """
        Saves the results to file.

        Parameters
        ----------
            data (pd.DataFrame): Data to save.
            subfolder (str): Subfolder to save the results.
            exp_type (str): Experiment type.
            exp_group (str): Experiment group.
            save_format (str): File type.
            distance_type (str): Distance type used in experiment.
            should_save_time (bool, default=True): If timestamp should be saved.
        Returns
        ----------
            Timestamp or None
        """

        time_str = time.strftime('%Y%m%d_%H%M%S') if should_save_time else ""
        output_dir = os.path.join(self.base_dir, exp_group, f"{time_str}", subfolder)
        
        os.makedirs(output_dir, exist_ok=True)

        if distance_type is not None:
            path = os.path.join(output_dir, f"{exp_type}_{distance_type}")
        else:
            path = os.path.join(output_dir, exp_type)

        self.write_to_disk(data, path, save_format)

        return time_str
       

# One time initialization of serializer object
serializer = Serialization()
