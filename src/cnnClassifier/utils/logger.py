import os
import sys
from loguru import logger

def configure_logger(
    logger_name=None,
    log_dir="logs",
    log_file="running_logs.log",
    level="DEBUG",
    log_format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level><bold>{level: <8}</bold></level> | "
        "<blue>{name}</blue>:<yellow>{function}</yellow>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
):
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_file)

    logger.remove()

    # Use outro nome para a variável local
    if logger_name:
        log = logger.bind(name=logger_name)
    else:
        log = logger.bind(name="root")

    log.add(log_filepath, level=level, format=log_format, enqueue=True)
    log.add(sys.stdout, level=level, format=log_format)
    return log

logger_instance = configure_logger(logger_name="cnnClassifier")

logger = logger_instance
