import logging
import os
import time


class Logger:
    """
    A simple logging utility class that configures a logger to output messages
    to both the console and a file.

    Parameters
    ----------
        name: str
            Name of the project.
        logdir: str
            Directory to save logs.
        level: int
            Logging level
    """

    def __init__(self, name = "SuperResponder", log_dir = "./logs", level = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        os.makedirs(log_dir, exist_ok = True)
        
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


    def critical(self, message):
        self.logger.critical(message)


# Singleton
logger = Logger()
