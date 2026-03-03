import os
import time
import json
import joblib
from typing import Literal # Keep this, it's used
import pandas as pd

from logger import logger
from utils.utils import _sanitize_filename

class Serialization:
    """
    Serialize test results to files.
    
    Parameters
    ----------
        base_dir: str
            Directory to store results
        cache_dir: str
            Directory to store cached datasets in feather format.
        plot_dir: str
            Directory to store plots.
    """

    def __init__(self, base_dir = "./results", cache_dir = ".cached_datasets", plot_dir = "./plots"):
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.logger = logger
        self.plot_dir = plot_dir


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
                      data,
                      path, 
                      save_format,
                      index = True):
        """
        Writes results to file.
        """
        
        self.logger.info(f"Saving results to file: {path}.{save_format}")

        try:            
            if save_format in ["csv", "tsv"]:
                separator = "," if save_format == "csv" else "\t"
                data.to_csv(f"{path}.{save_format}", sep = separator, index = index)
            
            elif save_format == "xlsx":
                data.to_excel(f"{path}.{save_format}", index = index)
            
            elif save_format == "json":
                if not isinstance(data, dict):
                    self.logger.error("Cannot save non-dict data as JSON")
                    return
                
                json.dump(data, open(f"{path}.{save_format}", "w"), indent = 4)
            
            elif save_format == "pkl":
                joblib.dump(data, f"{path}.{save_format}")
            
            else:
                self.logger.error(f"Unsupported file format: {save_format}")
                raise ValueError()
            
        except Exception as e:                                                                                          
            self.logger.error(f"Error saving file: {e}")


    def save_file(self, 
                  data, 
                  subfolder: str, 
                  exp_type: str, 
                  exp_group: str, 
                  save_format: Literal['csv', 'tsv', 'xlsx', 'json', 'pkl'],
                  distance_type: str = None,
                  should_save_time = True) -> str | None:
        """
        Saves the results to file.

        Parameters
        ----------
            data: pd.DataFrame or dict or trained model
                Data to save.
            subfolder: str
                Subfolder to save the results.
            exp_type: str
                Experiment type.
            exp_group: str
                Experiment group.
            save_format: str
                File type. Supported file types are csv, tsv, xlsx, for pandas dataframes,
                json for dicts and pkl for trained models.
            distance_type: str
                Distance type used in experiment.
            should_save_time: bool, default=True
                If timestamp should be saved.
        Returns
        ----------
            Timestamp or None
        """

        time_str = time.strftime('%Y%m%d_%H%M%S') if should_save_time else ""
        output_dir = os.path.join(self.base_dir, exp_group, f"{time_str}", subfolder)
        
        os.makedirs(output_dir, exist_ok = True)

        if distance_type is not None:
            path = os.path.join(output_dir, f"{exp_type}_{distance_type}")
        else:
            path = os.path.join(output_dir, exp_type)

        self.write_to_disk(data, path, save_format)

        return time_str
       
    
    def save_plot(self, fig, subfolder, exp_name, metric_name, bbox_inches = None):
        """
        Saves the plot to a file with metric-specific naming.
        """

        out_path = os.path.join(self.plot_dir, subfolder, f"{time.strftime('%Y%m%d_%H%M%S')}", exp_name)
        os.makedirs(out_path, exist_ok = True)
        
        safe_name = _sanitize_filename(metric_name)
        dest_name = f"{exp_name}_{safe_name}" if exp_name != safe_name else f"{safe_name}"
        save_path = os.path.join(out_path, f"{dest_name}.png")
        
        try: 
            fig.savefig(save_path, dpi = 300, bbox_inches = bbox_inches) 
            self.logger.info(f"Plot saved to: {save_path}")
        
        except Exception as e:
            self.logger.warning(f"Error saving plot to: {save_path}: {e}")
