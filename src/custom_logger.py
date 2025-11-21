import logging
import os

def get_custom_logger(name=__name__, level=logging.DEBUG, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    
    formatter = logging.Formatter("{asctime} - {levelname} - {name} - {message}"
                                , style="{"
                                , datefmt="%Y-%m-%d %H:%M")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger