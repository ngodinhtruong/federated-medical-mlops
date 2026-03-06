import time
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from zoneinfo import ZoneInfo

from utils.logger_utils import logger

from flwr.common import (
    FitIns,
    GetParametersIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from utils.minio_utils import (
    minio_put_json,
    minio_put_bytes,
    fl_params_to_pth_bytes,
    get_eligible_clients,
)

from utils.fl_utils import (
    call_fit,
    call_get_parameters,
)

from utils.model_utils  import (
    server_eval,
    server_val_loss,
)
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


run_ts = datetime.now(VN_TZ).strftime("%Y-%m-%d_%H-%M-%S")

BUCKET = os.getenv("MINIO_BUCKET", "fl-artifacts")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "cycles")
SERVER_ID = os.getenv("SERVER_ID", "server-1")
CYCLE_ID = os.getenv("CYCLE_ID", run_ts)

class AOFLAsyncServer:
    def __init__(self,client_manager,val_X,val_y,*,alpha0=0.5,max_updates=200,concurrency=3,max_rounds_per_cycle=10,Log_file=None, minio_client=None):
        self.client_manager = client_manager
        self.val_X = val_X
        self.val_y = val_y
        self.log_file = Log_file
        self.alpha0 = float(alpha0)
        self.max_updates = int(max_updates)
        self.concurrency = int(concurrency)
        self.max_rounds_per_cycle = int(max_rounds_per_cycle)

        self.version = 0
        self.global_parameters = None
        self.latest_parameters = None

        self.best_loss = float("inf")
        self.best_parameters = None
        self.best_round = None

        self.cid_to_client_id = {}

        self._eligible_cache = set()
        self._eligible_cache_ts = 0.0
        self.eligibility_ttl_sec = float(os.getenv("AOFL_ELIG_TTL_SEC", "2.0"))

        self.idle_timeout_sec = float(os.getenv("AOFL_IDLE_TIMEOUT_SEC", "30.0"))

        self.cycle_prefix = f"{MINIO_PREFIX}/cycle_{CYCLE_ID}"
        self.round_metrics = []
        self.minio_client = minio_client
        logger.info(f"Online Fedarated Learning started: (alpha0={self.alpha0})")
        logger.info(
            f"[AOFL-ASYNC] concurrency={self.concurrency} "
            f"max_updates={self.max_updates} max_rounds_per_cycle={self.max_rounds_per_cycle}"
        )
        logger.info(
            f"[AOFL-ASYNC] elig_ttl={self.eligibility_ttl_sec}s idle_timeout={self.idle_timeout_sec}s"
        )
        logger.info(f"[MINIO] bucket={BUCKET} prefix={self.cycle_prefix}")

    def _refresh_eligible(self):
        now = time.time()
        if now - self._eligible_cache_ts < self.eligibility_ttl_sec:
            return self._eligible_cache

        eligible = set(get_eligible_clients(self.minio_client, BUCKET))
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
        self.best_round = 0
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

    def _upload_cycle_artifacts(self,*,meta,summary,best_params,last_params,round_metrics):
        minio_put_json(self.minio_client, BUCKET, f"{self.cycle_prefix}/meta.json", meta)

        minio_put_json(self.minio_client, BUCKET, f"{self.cycle_prefix}/summary.json", summary)

        minio_put_json(self.minio_client, BUCKET, f"{self.cycle_prefix}/round_metrics.json", {"rounds": round_metrics})

        best_raw = fl_params_to_pth_bytes(best_params)
        last_raw = fl_params_to_pth_bytes(last_params)

        minio_put_bytes(self.minio_client, BUCKET, f"{self.cycle_prefix}/model_best.pth", best_raw, "application/octet-stream")
        minio_put_bytes(self.minio_client, BUCKET, f"{self.cycle_prefix}/model_last.pth", last_raw, "application/octet-stream")

        try:
            if self.log_file and os.path.exists(self.log_file):

                with open(self.log_file, "rb") as f:
                    log_raw = f.read()

                minio_put_bytes(
                    BUCKET,
                    f"{self.cycle_prefix}/server.log",
                    log_raw,
                    "text/plain",
                )

        except Exception as e:
            logger.info(f"[MINIO] Skip uploading log file: {e}")

        done = {
            "cycle_id": str(CYCLE_ID),
            "server_id": str(SERVER_ID),
            "status": "completed",
            "ts": datetime.now(VN_TZ).isoformat(),
        }
        minio_put_json(self.minio_client, BUCKET, f"{self.cycle_prefix}/DONE.json", done)

    def run_one_cycle(self):
        cycle_started_at = datetime.now(VN_TZ).isoformat()

        if self.global_parameters is None:
            self._get_initial_parameters()

        self.best_loss = float("inf")
        self.best_parameters = self.global_parameters
        self.best_round = None

        self.round_metrics = []

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

                self.global_parameters = ndarrays_to_parameters(global_nd)
                self.latest_parameters = self.global_parameters

                global_loss = server_val_loss(self.latest_parameters, self.val_X, self.val_y)

                record = {
                    "ts": datetime.now(VN_TZ).isoformat(),
                    "cycle_id": str(CYCLE_ID),
                    "round_id": int(rid),
                    "client_id": str(client_id),
                    "flower_cid": str(flower_cid),
                    "base_version": int(base_version),
                    "tau": int(tau),
                    "alpha": float(alpha),
                    "server_version_after": int(self.version),
                    "train_time": float(m.get("train_time", 0.0)),
                    "client_train_loss": float(m.get("train_loss", 0.0)) if m.get("train_loss") is not None else None,
                    "client_train_acc": float(m.get("train_acc", 0.0)) if m.get("train_acc") is not None else None,
                    "client_val_loss": float(m.get("val_loss", 0.0)) if m.get("val_loss") is not None else None,
                    "client_val_acc": float(m.get("val_acc", 0.0)) if m.get("val_acc") is not None else None,
                    "server_global_loss": float(global_loss),
                }
                self.round_metrics.append(record)

                logger.info(
                    f"[AOFL][ROUND {rid}][CLIENT {client_id}] "
                    f"train_time={record['train_time']:.3f}s base_version={base_version} "
                    f"-> tau={tau}, alpha={alpha:.4f}, server_version={self.version}"
                )
                logger.info(f"Round {rid} | global_loss={global_loss:.6f}")

                if global_loss < self.best_loss:
                    self.best_loss = global_loss
                    self.best_parameters = self.latest_parameters
                    self.best_round = int(rid)
                    logger.info(f"NEW BEST MODEL at round {rid} (loss={global_loss:.6f})")

                while len(inflight) < self.concurrency:
                    if not submit_one():
                        break

                break

        executor.shutdown(wait=True)

        last_parameters = self.latest_parameters
        best_parameters = self.best_parameters if self.best_parameters is not None else last_parameters

        best_eval = server_eval(best_parameters, self.val_X, self.val_y)
        last_eval = server_eval(last_parameters, self.val_X, self.val_y)

        logger.info("===== FINAL VALIDATION (BEST MODEL) =====")
        logger.info(f"Loss     : {best_eval['loss']:.6f}")
        logger.info(f"Accuracy : {best_eval['accuracy']:.6f}")
        logger.info(f"Precision: {best_eval['precision']:.6f}")
        logger.info(f"Recall   : {best_eval['recall']:.6f}")
        logger.info(f"F1-score : {best_eval['f1']:.6f}")

        logger.info("===== FINAL VALIDATION (LAST MODEL) =====")
        logger.info(f"Loss     : {last_eval['loss']:.6f}")
        logger.info(f"Accuracy : {last_eval['accuracy']:.6f}")
        logger.info(f"Precision: {last_eval['precision']:.6f}")
        logger.info(f"Recall   : {last_eval['recall']:.6f}")
        logger.info(f"F1-score : {last_eval['f1']:.6f}")

        logger.info(
            f"[AOFL-ASYNC] Cycle finished: rounds_dispatched={round_id}/{self.max_rounds_per_cycle} "
            f"updates_applied={updates_applied}"
        )

        meta = {
            "cycle_id": str(CYCLE_ID),
            "server_id": str(SERVER_ID),
            "run_ts": str(run_ts),
            "cycle_started_at": cycle_started_at,
            "cycle_finished_at": datetime.now(VN_TZ).isoformat(),
            "alpha0": float(self.alpha0),
            "max_updates": int(self.max_updates),
            "concurrency": int(self.concurrency),
            "max_rounds_per_cycle": int(self.max_rounds_per_cycle),
            "idle_timeout_sec": float(self.idle_timeout_sec),
            "eligibility_ttl_sec": float(self.eligibility_ttl_sec),
            "server_address": str(os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8080")),
        }

        summary = {
            "cycle_id": str(CYCLE_ID),
            "server_id": str(SERVER_ID),
            "rounds_dispatched": int(round_id),
            "updates_applied": int(updates_applied),
            "server_version_end": int(self.version),
            "best_round": int(self.best_round) if self.best_round is not None else None,
            "best": best_eval,
            "last": last_eval,
            "best_loss": float(best_eval["loss"]),
            "last_loss": float(last_eval["loss"]),
        }

        try:
            self._upload_cycle_artifacts(
                meta=meta,
                summary=summary,
                best_params=best_parameters,
                last_params=last_parameters,
                round_metrics=self.round_metrics,
            )
            logger.info(f"[MINIO] Uploaded artifacts to {BUCKET}/{self.cycle_prefix}")
        except Exception as e:
            logger.exception(f"[MINIO] Upload failed: {e}")