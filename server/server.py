# server_aofl_async.py
import os
import io
import json
import time
import random
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import (
    FitIns,
    GetParametersIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from model.MLP import MLP
from data.load_data import load_test_ds
from minio import Minio


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

LOG_DIR = "/opt/fl/logs"
os.makedirs(LOG_DIR, exist_ok=True)

run_ts = datetime.now(VN_TZ).strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"fl_server_{run_ts}.log")

logger = logging.getLogger("fl_server")
logger.setLevel(logging.INFO)


class VNFormatter(logging.Formatter):
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, VN_TZ).timetuple()


formatter = VNFormatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info(f"Logging initialized → {LOG_FILE}")


MINIO_ENDPOINT = "minio:9000"
MINIO_USER = "admin"
MINIO_PASS = "admin12345"
BUCKET = "fl-artifacts"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_USER,
    secret_key=MINIO_PASS,
    secure=False,
)


def get_eligible_clients():
    date = datetime.utcnow().strftime("%Y-%m-%d")
    prefix = f"status/{date}/"

    eligible = []
    for obj in minio_client.list_objects(BUCKET, prefix=prefix, recursive=True):
        raw = minio_client.get_object(BUCKET, obj.object_name).read()
        data = json.loads(raw)
        if data.get("has_data") and data.get("can_train"):
            eligible.append(str(data.get("client_id")))
    return eligible


def binary_metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    eps = 1e-8
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return accuracy, precision, recall, f1


def call_get_parameters(cp, ins):
    return cp.get_parameters(ins, timeout=None)


def call_fit(cp, fit_ins):
    return cp.fit(fit_ins, timeout=None)


def start_flower_grpc_server(server_address: str):
    try:
        from flwr.server.client_manager import SimpleClientManager
        from flwr.server.grpc_server import start_grpc_server
    except Exception:
        from flwr.server.client_manager import SimpleClientManager  # type: ignore
        from flwr.server.app import start_grpc_server  # type: ignore

    client_manager = SimpleClientManager()

    grpc_server = start_grpc_server(
        server_address=server_address,
        client_manager=client_manager,
        max_message_length=536870912,
    )
    return grpc_server, client_manager


def build_model_from_parameters(params, input_shape):
    flat_dim = int(np.prod(input_shape))
    model = MLP(input_dim=flat_dim)
    model.eval()

    weights = fl.common.parameters_to_ndarrays(params)
    state_dict = model.state_dict()
    for k, w in zip(state_dict.keys(), weights):
        state_dict[k] = torch.tensor(w)
    model.load_state_dict(state_dict)
    return model


def server_val_loss(params, val_X, val_y):
    model = build_model_from_parameters(params, val_X.shape[1:])
    loss_fn = nn.BCELoss()

    with torch.no_grad():
        X = val_X.view(val_X.size(0), -1)
        pred = model(X)
        loss = loss_fn(pred, val_y.float())
    return float(loss.item())


class AOFLAsyncServer:
    def __init__(
        self,
        client_manager,
        val_X,
        val_y,
        *,
        alpha0=0.5,
        max_updates=200,
        concurrency=3,
        max_rounds_per_cycle=10,
    ):
        self.client_manager = client_manager
        self.val_X = val_X
        self.val_y = val_y

        self.alpha0 = float(alpha0)
        self.max_updates = int(max_updates)
        self.concurrency = int(concurrency)
        self.max_rounds_per_cycle = int(max_rounds_per_cycle)

        self.version = 0
        self.global_parameters = None
        self.latest_parameters = None

        self.best_loss = float("inf")
        self.best_parameters = None

        self.cid_to_client_id = {}

        self._eligible_cache = set()
        self._eligible_cache_ts = 0.0
        self.eligibility_ttl_sec = float(os.getenv("AOFL_ELIG_TTL_SEC", "2.0"))

        self.idle_timeout_sec = float(os.getenv("AOFL_IDLE_TIMEOUT_SEC", "30.0"))

        logger.info(f"Online Fedarated Learning started: (alpha0={self.alpha0})")
        logger.info(
            f"[AOFL-ASYNC] concurrency={self.concurrency} "
            f"max_updates={self.max_updates} max_rounds_per_cycle={self.max_rounds_per_cycle}"
        )
        logger.info(
            f"[AOFL-ASYNC] elig_ttl={self.eligibility_ttl_sec}s idle_timeout={self.idle_timeout_sec}s"
        )

    def _refresh_eligible(self):
        now = time.time()
        if now - self._eligible_cache_ts < self.eligibility_ttl_sec:
            return self._eligible_cache

        eligible = set(get_eligible_clients())
        self._eligible_cache = eligible
        self._eligible_cache_ts = now
        logger.info(f"[SERVER] Eligible clients (dynamic): {sorted(list(eligible))}")
        return eligible

    def _wait_for_clients(self, n=1, timeout_sec=120):
        t0 = time.time()
        while True:
            clients = list(self.client_manager.all().values())
            if len(clients) >= n:
                return clients
            if time.time() - t0 > timeout_sec:
                raise RuntimeError(f"Timeout waiting for {n} clients, currently {len(clients)}")
            time.sleep(0.5)

    def _get_initial_parameters(self):
        self._wait_for_clients(n=1, timeout_sec=120)
        clients = list(self.client_manager.all().values())
        cp = random.choice(clients)

        logger.info("[SERVER] Requesting initial parameters from a client...")
        ins = GetParametersIns(config={})
        res = call_get_parameters(cp, ins)

        self.global_parameters = res.parameters
        self.latest_parameters = self.global_parameters

        init_loss = server_val_loss(self.global_parameters, self.val_X, self.val_y)
        self.best_loss = init_loss
        self.best_parameters = self.global_parameters
        logger.info(f"[SERVER] Initial global_loss={init_loss:.6f}")

    def _dispatch_fit(self, cp, round_id, params_snapshot, base_version_snapshot):
        fit_ins = FitIns(
            parameters=params_snapshot,
            config={
                "server_round": round_id,
                "server_version": base_version_snapshot,
            },
        )
        fit_res = call_fit(cp, fit_ins)
        return cp, fit_res

    def run_one_cycle(self):
        if self.global_parameters is None:
            self._get_initial_parameters()

        self.best_loss = float("inf")
        self.best_parameters = self.global_parameters

        self._eligible_cache = set()
        self._eligible_cache_ts = 0.0
        self.cid_to_client_id = {}

        executor = ThreadPoolExecutor(max_workers=self.concurrency)

        inflight = {}
        busy_cids = set()
        round_id = 0
        updates_applied = 0
        last_progress_ts = time.time()

        def snapshot_parameters(params):
            return ndarrays_to_parameters(parameters_to_ndarrays(params))

        def pick_idle_client_for_dispatch():
            clients = list(self.client_manager.all().values())
            if not clients:
                return None

            eligible_now = self._refresh_eligible()

            eligible_idle = []
            unknown_idle = []

            for cp in clients:
                fc = str(getattr(cp, "cid", ""))
                if fc in busy_cids:
                    continue

                logical_id = self.cid_to_client_id.get(fc)
                if logical_id is None:
                    unknown_idle.append(cp)
                else:
                    if logical_id in eligible_now:
                        eligible_idle.append(cp)

            if eligible_idle:
                return random.choice(eligible_idle)

            if unknown_idle and eligible_now:
                return random.choice(unknown_idle)

            return None

        def submit_one():
            nonlocal round_id

            if round_id >= self.max_rounds_per_cycle:
                return False

            cp = pick_idle_client_for_dispatch()
            if cp is None:
                return False

            fc = str(getattr(cp, "cid", ""))
            round_id += 1

            params_snapshot = snapshot_parameters(self.global_parameters)
            base_version_snapshot = int(self.version)

            fut = executor.submit(
                self._dispatch_fit,
                cp,
                round_id,
                params_snapshot,
                base_version_snapshot,
            )

            inflight[fut] = (round_id, base_version_snapshot, fc)
            busy_cids.add(fc)

            logger.info(
                f"[DISPATCH] round={round_id}/{self.max_rounds_per_cycle} -> flower_cid={fc} "
                f"base_version={base_version_snapshot} inflight={len(inflight)}/{self.concurrency}"
            )
            return True

        while len(inflight) < self.concurrency:
            if not submit_one():
                break

        while True:
            if round_id >= self.max_rounds_per_cycle and not inflight:
                logger.info(f"[AOFL-ASYNC] Reached max rounds per cycle: {self.max_rounds_per_cycle}")
                break

            if not inflight:
                if submit_one():
                    last_progress_ts = time.time()
                    continue

                if time.time() - last_progress_ts > self.idle_timeout_sec:
                    logger.info("[AOFL-ASYNC] No eligible training for too long -> end cycle")
                    break

                time.sleep(0.5)
                continue

            for fut in as_completed(list(inflight.keys()), timeout=None):
                rid, dispatched_base_version, fc = inflight.pop(fut)
                busy_cids.discard(fc)

                try:
                    cp, fit_res = fut.result()
                except Exception as e:
                    logger.exception(f"[AOFL][ROUND {rid}] Fit failed: {e}")
                    last_progress_ts = time.time()
                    break

                m = fit_res.metrics or {}
                client_id = str(m.get("client_id", "unknown"))

                flower_cid = str(getattr(cp, "cid", ""))
                self.cid_to_client_id[flower_cid] = client_id

                base_version = int(m.get("base_version", dispatched_base_version))
                cur_ver_before = int(self.version)

                logger.info(
                    f"[RECV] round={rid} <- client={client_id} flower_cid={flower_cid} "
                    f"base_version={base_version} server_version_before={cur_ver_before}"
                )

                logger.info(
                    f"[ROUND {rid}][CLIENT {client_id}] "
                    f"train_loss={m.get('train_loss'):.4f}, "
                    f"train_acc={m.get('train_acc'):.4f} | "
                    f"val_loss={m.get('val_loss'):.4f}, "
                    f"val_acc={m.get('val_acc'):.4f}"
                )

                logger.info(
                    f"[ROUND {rid}][GLOBAL] "
                    f"train_loss={m.get('train_loss'):.4f}, "
                    f"train_acc={m.get('train_acc'):.4f} | "
                    f"val_loss={m.get('val_loss'):.4f}, "
                    f"val_acc={m.get('val_acc'):.4f}"
                )

                eligible_now = self._refresh_eligible()
                if client_id not in eligible_now:
                    logger.info(f"[AOFL][ROUND {rid}] SKIP update because client {client_id} is NOT eligible now")
                    last_progress_ts = time.time()
                    while len(inflight) < self.concurrency:
                        if not submit_one():
                            break
                    break

                tau = max(0, self.version - base_version)
                alpha = self.alpha0 / (1 + tau)

                global_nd = parameters_to_ndarrays(self.global_parameters)
                client_nd = parameters_to_ndarrays(fit_res.parameters)
                global_nd = [(1 - alpha) * g + alpha * c for g, c in zip(global_nd, client_nd)]

                self.version += 1
                updates_applied += 1
                last_progress_ts = time.time()

                tt = float(m.get("train_time", 0.0))
                logger.info(
                    f"[AOFL][ROUND {rid}][CLIENT {client_id}] "
                    f"train_time={tt:.3f}s base_version={base_version} "
                    f"-> tau={tau}, alpha={alpha:.4f}, server_version={self.version}"
                )

                self.global_parameters = ndarrays_to_parameters(global_nd)
                self.latest_parameters = self.global_parameters

                loss = server_val_loss(self.latest_parameters, self.val_X, self.val_y)
                logger.info(f"Round {rid} | global_loss={loss:.6f}")

                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_parameters = self.latest_parameters
                    logger.info(f"NEW BEST MODEL at round {rid} (loss={loss:.6f})")

                while len(inflight) < self.concurrency:
                    if not submit_one():
                        break

                break

        executor.shutdown(wait=True)

        logger.info("===== FINAL VALIDATION (BEST MODEL) =====")

        model = build_model_from_parameters(self.best_parameters, self.val_X.shape[1:])

        with torch.no_grad():
            X = self.val_X.view(self.val_X.size(0), -1)
            logits = model(X)
            preds = (logits > 0.5).int().cpu().numpy()

        y_true = self.val_y.cpu().numpy()
        y_pred = preds

        acc, prec, rec, f1 = binary_metrics(y_true, y_pred)

        logger.info("FINAL METRICS")
        logger.info(f"Accuracy : {acc:.6f}")
        logger.info(f"Precision: {prec:.6f}")
        logger.info(f"Recall   : {rec:.6f}")
        logger.info(f"F1-score : {f1:.6f}")

        logger.info(
            f"[AOFL-ASYNC] Cycle finished: rounds_dispatched={round_id}/{self.max_rounds_per_cycle} "
            f"updates_applied={updates_applied}"
        )


def stage_train(aofl):
    aofl.run_one_cycle()


def stage_next(client_manager):
    time.sleep(1.0)


def main():
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
    )

    stage = os.getenv("PIPELINE_STAGE", "train")

    try:
        if stage == "train":
            stage_train(aofl)
            stage = "next"
            logger.info("[PIPELINE] Switch stage -> next")

        if stage == "next":
            stage_next(client_manager)
            logger.info("[PIPELINE] Next stage done")
    finally:
        grpc_server.stop(grace=None)
        logger.info("[SERVER] gRPC stopped")


if __name__ == "__main__":
    main()
