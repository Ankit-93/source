import logging
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),'logs',LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s -%(message)s",
#     level=logging.INFO
# )
def get_logger(name, level=logging.INFO, log_file=None, console=True):
    """
    Creates and configures a logger.

    :param name: Name of the logger.
    :param level: Logging level (default: logging.INFO).
    :param log_file: Path to the log file (default: None, logs will not be saved to a file).
    :param console: Boolean flag to log to console (default: True).
    :return: Configured logger.
    """
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    handlers = []
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10**6, backupCount=3)
        handlers.append(file_handler)
    if console:
        console_handler = logging.StreamHandler()
        handlers.append(console_handler)
    
    # Set logging level and format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = get_logger(name="appLog", level=logging.DEBUG, log_file="app.log", console=True)

# if __name__=="__main__":

#     logging.info("Logging has started...")