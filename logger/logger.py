import logging
import os
import time
from sklearn.preprocessing import OrdinalEncoder 

class Logger:
    """
    A simple logging utility class that configures a logger to output messages
    to both the console and a file.
    """
    def __init__(self, name="SuperResponder", log_dir="./logs", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Formatter for console and file
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Add handlers only if they haven't been added before to prevent duplicate logs
        if not self.logger.handlers:
            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # File handler
            log_filename = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
            fh = logging.FileHandler(log_filename)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)


def format_pipeline(pipeline, name):
    """
    Helper method to format a Scikit-learn Pipeline for readable logging.
    """
    pipeline_str = f"  - {name} pipeline:\n"
    for step_name, step_estimator in pipeline.steps:
        pipeline_str += f"    - {step_name}: {repr(step_estimator)}\n"
    return pipeline_str

# Create a single, module-level instance of the Logger
logger = Logger()
