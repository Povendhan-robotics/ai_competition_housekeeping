"""Logging, timers, and device helpers."""
import time
import logging
import torch

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
