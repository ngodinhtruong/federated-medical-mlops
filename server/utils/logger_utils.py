import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

logger = logging.getLogger("fl_server")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    console.setFormatter(formatter)

    logger.addHandler(console)


def enable_file_logging(log_dir="/opt/fl/logs"):
    """
    Enable file logging only when training starts
    """

    os.makedirs(log_dir, exist_ok=True)

    run_ts = datetime.now(VN_TZ).strftime("%Y-%m-%d_%H-%M-%S")

    log_file = os.path.join(log_dir, f"fl_server_{run_ts}.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info(f"File logging enabled → {log_file}")

    return log_file