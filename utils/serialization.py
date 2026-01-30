import os
import time
import pandas as pd

from logger.logger import logger


class Serialization:
    """
    Serialize test results to files.
    
    Parameters
    ----------
        base_dir (str): Base directory to save results.
    """

    def __init__(self, base_dir="./results"):
        self.base_dir = base_dir
        self.logger = logger
        os.makedirs(self.base_dir, exist_ok=True)    


    def save_file(self, data: pd.DataFrame, subfolder, exp_type, exp_group, 
                  save_format, distance_type=None, should_save_time=True) -> str | None:
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

        path = os.path.join(output_dir, f"{exp_type}_{distance_type}.{save_format}")
        
        self.logger.info(f"Saving results to file: {path}")

        separator = "," if save_format == "csv" else "\t"

        try:
            if save_format in ["csv", "tsv"]:
                data.to_csv(path, sep=separator)
            
            elif save_format == "xlsx":
                data.to_excel(path)
            
            return time_str
        
        except Exception as e:                                                                                          
            self.logger.error(f"Error saving file: {e}")


# One time initialization of serializer object
serializer = Serialization()

if __name__ == "__main__":

    serializer = Serialization()
    serializer.save_to_excel(None, "test", "euclidean")  # Example usage