from datetime import datetime
import os
import time
from zoneinfo import ZoneInfo
from minio import Minio

from data.load_data import load_test_ds

from core_flower.aofl_async_server import AOFLAsyncServer, logger
from utils.logger_utils import logger, enable_file_logging

from utils.fl_utils import start_flower_grpc_server

def stage_train(aofl):
    log_file = enable_file_logging()

    aofl.log_file = log_file
    aofl.run_one_cycle()

def main(minio_client):
    logger.info("Starting Flower server (CUSTOM AOFL ASYNC - EVENT ELIGIBLE)")

    val_X, val_y = load_test_ds()
    val_X = val_X.float()
    val_y = val_y.int()

    server_address = os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8080")
    grpc_server, client_manager = start_flower_grpc_server(server_address)
    logger.info(f"[SERVER] gRPC listening at {server_address}")

    alpha0 = float(os.getenv("AOFL_ALPHA0", "0.5"))
    max_updates = int(os.getenv("AOFL_MAX_UPDATES", "200"))
    concurrency = int(os.getenv("AOFL_CONCURRENCY", "3"))
    max_rounds_per_cycle = int(os.getenv("AOFL_MAX_ROUNDS_PER_CYCLE", "10"))

    aofl = AOFLAsyncServer(
        client_manager=client_manager,
        val_X=val_X,
        val_y=val_y,
        alpha0=alpha0,
        max_updates=max_updates,
        concurrency=concurrency,
        max_rounds_per_cycle=max_rounds_per_cycle,
        minio_client=minio_client,
    )

    stage = os.getenv("PIPELINE_STAGE", "train")

    try:
        if stage == "train":
            stage_train(aofl)
            stage = "next"
            run_ts = datetime.now(VN_TZ).strftime("%Y-%m-%d_%H-%M-%S")
            logger.info(f"Run timestamp: {run_ts}")
            logger.info("[PIPELINE] Switch stage -> next")

        if stage == "next":
            time.sleep(1.0)
            logger.info("[PIPELINE] Next stage done")
    finally:
        grpc_server.stop(grace=None)
        logger.info("[SERVER] gRPC stopped")


if __name__ == "__main__":

    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_USER = os.getenv("MINIO_ACCESS_KEY", "admin")
    MINIO_PASS = os.getenv("MINIO_SECRET_KEY", "admin12345")
    VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

    


    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_USER,
        secret_key=MINIO_PASS,
        secure=False,
    )
    main(minio_client)