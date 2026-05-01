import os
import yaml
import math
import torch
import logging


_LOGGER = None


def read_configuration(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def init_logger(args):
    global _LOGGER

    log_dir = args.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    log_name = args.get("log_name", None)
    if log_name is None:
        base = args.get("model_name", "cet_mae")
        folder = args.get("folder_name", "default")
        log_name = f"{base}_{folder}.log"

    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger("cet_mae")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Logger initialized.")
    logger.info(f"Logging to: {log_path}")
    logger.info(f"Args/config: {args}")

    _LOGGER = logger


def getLogger():
    global _LOGGER
    if _LOGGER is None:
        init_logger({})
    return _LOGGER


class EarlyStopper:
    def __init__(self, patience=4, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None

    def early_stop(self, value):
        if self.best is None:
            self.best = value
            return False

        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def check_nan_inf(x, name="tensor"):
    if isinstance(x, torch.Tensor):
        if torch.isnan(x).any():
            raise ValueError(f"{name} contains NaN")
        if torch.isinf(x).any():
            raise ValueError(f"{name} contains Inf")
    return False
