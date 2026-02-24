import os
import json
import tempfile
from typing import Dict, Any, Optional

import mlflow
from minio import Minio
from airflow.exceptions import AirflowSkipException


def _minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "admin12345")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _get_json(client: Minio, bucket: str, object_name: str) -> Dict[str, Any]:
    resp = client.get_object(bucket, object_name)
    try:
        return json.loads(resp.read().decode("utf-8"))
    finally:
        try:
            resp.close()
        except Exception:
            pass
        try:
            resp.release_conn()
        except Exception:
            pass


def _get_bytes(client: Minio, bucket: str, object_name: str) -> bytes:
    resp = client.get_object(bucket, object_name)
    try:
        return resp.read()
    finally:
        try:
            resp.close()
        except Exception:
            pass
        try:
            resp.release_conn()
        except Exception:
            pass


def _exists(client: Minio, bucket: str, object_name: str) -> bool:
    try:
        client.stat_object(bucket, object_name)
        return True
    except Exception:
        return False


def _log_json_artifact(data: Dict[str, Any], artifact_path: str, filename: str):
    raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, filename)
        with open(p, "wb") as f:
            f.write(raw)
        mlflow.log_artifact(p, artifact_path=artifact_path)


def _log_bytes_artifact(raw: bytes, artifact_path: str, filename: str):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, filename)
        with open(p, "wb") as f:
            f.write(raw)
        mlflow.log_artifact(p, artifact_path=artifact_path)


def _log_metrics(prefix: str, metrics: Dict[str, Any]):
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"{prefix}{k}", float(v))


def log_cycle_to_mlflow(cycle_path: str, cycle_id: str):
    bucket = os.getenv("MINIO_BUCKET", "fl-artifacts")
    client = _minio_client()

    chosen_obj = f"{cycle_path}/CHOSEN.json"
    if not _exists(client, bucket, chosen_obj):
        raise AirflowSkipException("CHOSEN.json not ready")

    chosen_data = _get_json(client, bucket, chosen_obj)

    threshold = float(chosen_data.get("threshold", 0.5))
    chosen = str(chosen_data.get("chosen", "unknown"))
    criteria = str(chosen_data.get("criteria", "unknown"))
    chosen_model_object = chosen_data.get("chosen_object")

    meta = chosen_data.get("meta", {}) or {}
    summary = chosen_data.get("summary", {}) or {}

    best_metrics = chosen_data.get("best", {}) or {}
    last_metrics = chosen_data.get("last", {}) or {}
    chosen_metrics = best_metrics if chosen == "best" else last_metrics

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "fl_experiment"))

    run_name = f"cycle_{cycle_id}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("cycle_id", str(cycle_id))
        mlflow.set_tag("server_id", str(meta.get("server_id", "")))
        mlflow.set_tag("chosen", chosen)
        mlflow.set_tag("criteria", criteria)

        server_addr = meta.get("server_address")
        if server_addr is not None:
            mlflow.set_tag("server_address", str(server_addr))

        mlflow.log_param("threshold", threshold)

        for p in [
            "alpha0",
            "max_updates",
            "concurrency",
            "max_rounds_per_cycle",
            "idle_timeout_sec",
            "eligibility_ttl_sec",
        ]:
            if p in meta:
                mlflow.log_param(p, meta[p])

        for p in [
            "rounds_dispatched",
            "updates_applied",
            "server_version_end",
            "best_round",
            "best_loss",
            "last_loss",
        ]:
            if p in summary:
                mlflow.log_param(p, summary[p])

        _log_metrics("best_", best_metrics)
        _log_metrics("last_", last_metrics)
        _log_metrics("chosen_", chosen_metrics)

        mlflow.log_metric("chosen_is_best", 1.0 if chosen == "best" else 0.0)

        _log_json_artifact(chosen_data, "minio", "CHOSEN.json")

        meta_obj = f"{cycle_path}/meta.json"
        if _exists(client, bucket, meta_obj):
            _log_json_artifact(_get_json(client, bucket, meta_obj), "minio", "meta.json")

        summary_obj = f"{cycle_path}/summary.json"
        if _exists(client, bucket, summary_obj):
            _log_json_artifact(_get_json(client, bucket, summary_obj), "minio", "summary.json")

        rounds_obj = f"{cycle_path}/round_metrics.json"
        if _exists(client, bucket, rounds_obj):
            _log_json_artifact(_get_json(client, bucket, rounds_obj), "minio", "round_metrics.json")

        server_log_obj = f"{cycle_path}/server.log"
        if _exists(client, bucket, server_log_obj):
            _log_bytes_artifact(_get_bytes(client, bucket, server_log_obj), "minio", "server.log")

        eval_best_obj = f"{cycle_path}/EVAL/eval_best.json"
        if _exists(client, bucket, eval_best_obj):
            _log_json_artifact(_get_json(client, bucket, eval_best_obj), "minio/EVAL", "eval_best.json")

        eval_last_obj = f"{cycle_path}/EVAL/eval_last.json"
        if _exists(client, bucket, eval_last_obj):
            _log_json_artifact(_get_json(client, bucket, eval_last_obj), "minio/EVAL", "eval_last.json")

        model_name = os.getenv("MLFLOW_MODEL_NAME", "fl_model")
        if chosen_model_object:
            raw = _get_bytes(client, bucket, str(chosen_model_object))
            with tempfile.TemporaryDirectory() as td:
                model_file = os.path.join(td, os.path.basename(str(chosen_model_object)))
                with open(model_file, "wb") as f:
                    f.write(raw)

                mlflow.log_artifact(model_file, artifact_path="model")

                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model"
                mlflow.register_model(model_uri, model_name)
        else:
            mlflow.set_tag("model_registered", "false")