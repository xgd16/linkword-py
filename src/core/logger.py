"""全局日志模块，可在任意模块中 from src.core.logger import logger 调用"""
import logging
import sys

from src.core.config import load_config

# 默认配置
_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _setup_logger() -> logging.Logger:
    """根据 config.yaml 初始化并返回 logger"""
    logger = logging.getLogger("py_server")
    if logger.handlers:
        return logger

    try:
        cfg = load_config()
        log_cfg = cfg.get("log", {})
        level = log_cfg.get("level", _DEFAULT_LEVEL)
        fmt = log_cfg.get("format", _DEFAULT_FORMAT)
        date_fmt = log_cfg.get("date_format", _DEFAULT_DATE_FORMAT)
    except Exception:
        level = _DEFAULT_LEVEL
        fmt = _DEFAULT_FORMAT
        date_fmt = _DEFAULT_DATE_FORMAT

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)
    formatter = logging.Formatter(fmt, datefmt=date_fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _setup_logger()
