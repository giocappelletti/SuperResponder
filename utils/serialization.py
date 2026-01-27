import os
import time

from logger.logger import logger


class Serialization:
    """
    Serialize test results to files.
    """

    def __init__(self, base_dir="./results"):
        self.base_dir = base_dir
        self.logger = logger
        os.makedirs(self.base_dir, exist_ok=True)

    @staticmethod
    def sanitize_filename(name):
        safe_name = name.lower().replace(" ", "_")
        return safe_name 

    def save_file(self, data, subfolder, exp_type, exp_group, save_format, distance_type=None, should_save_time=True):
        """Saves the results to file."""
        time_str = time.strftime('%Y%m%d_%H%M%S') if should_save_time else ""
        output_dir = os.path.join(self.base_dir, exp_group, f"{time_str}", subfolder)
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, 
                            f"{exp_type}_{distance_type}.{save_format}")
        
        self.logger.info(f"Saving results to file: {path}")

        try:
            if save_format == "csv" :
                data.to_csv(path)
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