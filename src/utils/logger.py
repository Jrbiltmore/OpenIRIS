
import logging

def setup_logger(name, log_file, level=logging.INFO):
    """
    Setup a logger with the given name and log file.
    Args:
    - name (str): Logger name.
    - log_file (str): Log file path.
    - level (logging.level): Logging level.
    Returns:
    - logger (logging.Logger): Configured logger.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger
