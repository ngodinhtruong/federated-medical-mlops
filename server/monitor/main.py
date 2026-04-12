"""
FL Monitor — FastAPI Backend
Đọc dữ liệu từ MinIO và cung cấp REST API cho Dashboard.
"""
import os
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from minio import Minio

app = FastAPI(title="FL Monitor")

# ==================== MINIO CONFIG ====================

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin12345")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET = os.getenv("MINIO_BUCKET", "fl-artifacts")

MODEL_PREFIXES = {
    "mlp": "training/mlp",
    "cnn": "training/cnn",
    "logreg": "training/logreg",
}


def mc():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)


def safe_get_json(client, bucket, key):
    try:
        resp = client.get_object(bucket, key)
        data = json.loads(resp.read().decode("utf-8"))
        resp.close()
        resp.release_conn()
        return data
    except Exception:
        return None


def list_objects_with_prefix(client, bucket, prefix, suffix=""):
    results = []
    try:
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            if suffix and not obj.object_name.endswith(suffix):
                continue
            results.append(obj.object_name)
    except Exception:
        pass
    return results


# ==================== API ENDPOINTS ====================

@app.get("/api/overview")
def get_overview():
    client = mc()

    # 1. Count clients
    today = datetime.utcnow().strftime("%Y-%m-%d")
    status_objs = list_objects_with_prefix(client, BUCKET, "clients/status/", ".json")
    client_ids = set()
    for obj in status_objs:
        parts = obj.split("/")
        if len(parts) >= 4:
            client_ids.add(parts[-1].replace(".json", ""))

    # 2. Champion
    champion = safe_get_json(client, BUCKET, "registry/champion.json")

    # 3. Count completed cycles
    total_cycles = 0
    for model, prefix in MODEL_PREFIXES.items():
        dones = list_objects_with_prefix(client, BUCKET, prefix + "/", "DONE.json")
        total_cycles += len(dones)

    # 4. System status
    status = "idle"
    if champion:
        status = "ready"

    return {
        "active_clients": len(client_ids),
        "client_ids": sorted(list(client_ids)),
        "champion": champion,
        "total_cycles": total_cycles,
        "system_status": status,
    }


@app.get("/api/clients")
def get_clients():
    client = mc()
    status_objs = list_objects_with_prefix(client, BUCKET, "clients/status/", ".json")

    clients_data = {}
    for obj in status_objs:
        data = safe_get_json(client, BUCKET, obj)
        if data and "client_id" in data:
            cid = data["client_id"]
            # Keep the latest status per client
            if cid not in clients_data or data.get("timestamp", "") > clients_data[cid].get("timestamp", ""):
                clients_data[cid] = data
                clients_data[cid]["status_path"] = obj

    # Add data quality info
    for cid in clients_data:
        dq_objs = list_objects_with_prefix(client, BUCKET, f"clients/data-quality/{cid}/", ".json")
        if dq_objs:
            dq_objs.sort(reverse=True)
            latest_dq = safe_get_json(client, BUCKET, dq_objs[0])
            clients_data[cid]["data_quality"] = latest_dq
        else:
            clients_data[cid]["data_quality"] = None

    return {"clients": list(clients_data.values())}


@app.get("/api/champion")
def get_champion():
    client = mc()
    champion = safe_get_json(client, BUCKET, "registry/champion.json")
    return {"champion": champion}


@app.get("/api/challenges")
def get_challenges():
    client = mc()
    objs = list_objects_with_prefix(client, BUCKET, "registry/challenges/", ".json")
    challenges = []
    for obj in objs:
        data = safe_get_json(client, BUCKET, obj)
        if data:
            data["_path"] = obj
            challenges.append(data)
    challenges.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return {"challenges": challenges}


@app.get("/api/cycles")
def get_cycles(model_type: str = Query(default="all")):
    client = mc()

    if model_type == "all":
        prefixes = MODEL_PREFIXES
    else:
        key = model_type.lower().replace("logisticregression", "logreg")
        if key in MODEL_PREFIXES:
            prefixes = {key: MODEL_PREFIXES[key]}
        else:
            return {"cycles": []}

    cycles = []
    for mtype, prefix in prefixes.items():
        done_objs = list_objects_with_prefix(client, BUCKET, prefix + "/", "DONE.json")
        for done_obj in done_objs:
            cycle_path = done_obj.replace("/DONE.json", "")
            cycle_id = cycle_path.split("/")[-1].replace("cycle_", "", 1)

            # Try to read CHOSEN.json
            chosen = safe_get_json(client, BUCKET, f"{cycle_path}/CHOSEN.json")
            eval_best = safe_get_json(client, BUCKET, f"{cycle_path}/EVAL/eval_best.json")
            eval_last = safe_get_json(client, BUCKET, f"{cycle_path}/EVAL/eval_last.json")

            # Check if processed
            processed = safe_get_json(client, BUCKET, f"{cycle_path}/PROCESSED.json")

            cycles.append({
                "model_type": mtype.upper() if mtype != "logreg" else "LogReg",
                "cycle_id": cycle_id,
                "cycle_path": cycle_path,
                "chosen": chosen,
                "eval_best": eval_best,
                "eval_last": eval_last,
                "processed": processed is not None,
            })

    cycles.sort(key=lambda x: x.get("cycle_id", ""), reverse=True)
    return {"cycles": cycles}


@app.get("/api/json")
def get_json_object(path: str = Query(...)):
    client = mc()
    data = safe_get_json(client, BUCKET, path)
    if data is None:
        return JSONResponse(status_code=404, content={"error": f"Object not found: {path}"})
    return {"path": path, "data": data}


# ==================== DOCKER LOGS & TASKS ====================

import docker

FL_CONTAINERS = ["fl-server-mlp", "fl-server-cnn", "fl-server-logreg"]

_dc_cache = None

def _docker_client():
    global _dc_cache
    if _dc_cache is not None:
        try:
            _dc_cache.ping()
            return _dc_cache
        except Exception:
            _dc_cache = None

    # Try multiple connection methods
    for url in [None, "unix:///var/run/docker.sock", "tcp://host.docker.internal:2375"]:
        try:
            if url is None:
                dc = docker.from_env()
            else:
                dc = docker.DockerClient(base_url=url)
            dc.ping()
            _dc_cache = dc
            return dc
        except Exception:
            continue
    return None


@app.get("/api/docker-test")
def docker_test():
    """Debug endpoint to check Docker connectivity."""
    dc = _docker_client()
    if not dc:
        return {"ok": False, "error": "Cannot connect to Docker"}
    try:
        containers = [c.name for c in dc.containers.list()]
        return {"ok": True, "containers": containers}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/logs/{container_name}")
def get_container_logs(container_name: str, tail: int = Query(default=80)):
    """Đọc N dòng log cuối cùng của 1 container Docker."""
    dc = _docker_client()
    if not dc:
        return {"error": "Docker not available", "logs": ""}
    try:
        container = dc.containers.get(container_name)
        logs = container.logs(tail=tail, timestamps=False).decode("utf-8", errors="replace")
        status = container.status  # running, exited, ...
        return {"container": container_name, "status": status, "logs": logs}
    except docker.errors.NotFound:
        return {"container": container_name, "status": "not_found", "logs": ""}
    except Exception as e:
        return {"container": container_name, "status": "error", "logs": str(e)}


@app.get("/api/server-logs")
def get_all_server_logs(tail: int = Query(default=50)):
    """Đọc log của cả 3 FL server cùng lúc."""
    results = {}
    dc = _docker_client()
    if not dc:
        return {"servers": {}}
    for name in FL_CONTAINERS:
        try:
            container = dc.containers.get(name)
            logs = container.logs(tail=tail, timestamps=False).decode("utf-8", errors="replace")
            results[name] = {"status": container.status, "logs": logs}
        except Exception:
            results[name] = {"status": "offline", "logs": ""}
    return {"servers": results}


@app.get("/api/running-tasks")
def get_running_tasks():
    """Phát hiện task nào đang chạy bằng cách quét Docker containers có tên chứa 'train_', 'ingest_', 'check_'."""
    dc = _docker_client()
    if not dc:
        return {"tasks": []}
    tasks = []
    try:
        for c in dc.containers.list():
            name = c.name
            # Airflow DockerOperator tạo container với tên chứa task_id
            for keyword in ["train_mlp", "train_cnn", "train_logistic", "train_logreg",
                            "ingest_new_data", "check_data_quality", "send_client_status"]:
                if keyword in name.lower():
                    # Tìm client ID từ env vars
                    env = {e.split("=")[0]: e.split("=", 1)[1] for e in (c.attrs.get("Config", {}).get("Env", []) or []) if "=" in e}
                    tasks.append({
                        "container": name,
                        "task": keyword,
                        "client_id": env.get("CLIENT_ID", "?"),
                        "model": env.get("MODEL_NAME", ""),
                        "status": c.status,
                        "started": c.attrs.get("State", {}).get("StartedAt", ""),
                    })
    except Exception:
        pass
    return {"tasks": tasks}


# ==================== SERVE STATIC ====================

app.mount("/", StaticFiles(directory="static", html=True), name="static")
