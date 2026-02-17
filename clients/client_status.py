import os
import time
import json
import io
from datetime import datetime
from minio import Minio


# =====================
# READ ENV
# =====================
def env_bool(key, default=False):
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


CLIENT_ID = os.getenv("CLIENT_ID")
if not CLIENT_ID:
    raise RuntimeError("CLIENT_ID is not set")

HAS_DATA = env_bool("CLIENT_HAS_DATA", True)
CAN_TRAIN = env_bool("CLIENT_CAN_TRAIN", True)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin12345")
BUCKET = os.getenv("MINIO_BUCKET", "fl-artifacts")


# =====================
# MINIO CLIENT
# =====================
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


# =====================
# HELPERS
# =====================
def ensure_bucket(retries=5):
    for i in range(retries):
        try:
            if not client.bucket_exists(BUCKET):
                print(f"[CLIENT-{CLIENT_ID}] Creating bucket {BUCKET}")
                client.make_bucket(BUCKET)
            return
        except Exception as e:
            print(f"[CLIENT-{CLIENT_ID}] Waiting for MinIO... ({i+1}) {e}")
            time.sleep(2)


def get_client_state_key():
    return f"cycle_state/{CLIENT_ID}.json"


def read_client_state():
    ensure_bucket()
    key = get_client_state_key()
    try:
        raw = client.get_object(BUCKET, key).read()
        data = json.loads(raw)
        if "cycle_id" not in data:
            data["cycle_id"] = 1
        if "data_version" not in data:
            data["data_version"] = None
        return data
    except Exception:
        return {"cycle_id": 1, "data_version": None}


def write_client_state(state):
    ensure_bucket()
    key = get_client_state_key()
    raw = json.dumps(state).encode("utf-8")
    client.put_object(
        BUCKET,
        key,
        io.BytesIO(raw),
        length=len(raw),
        content_type="application/json",
    )


def get_data_version():
    v = os.getenv("CLIENT_DATA_VERSION")
    if v is None or str(v).strip() == "":
        v = os.getenv("CLIENT_DATA_HASH")
    if v is None or str(v).strip() == "":
        v = None
    return v


def get_or_bump_cycle_id():
    state = read_client_state()
    prev_cycle_id = int(state.get("cycle_id", 1))
    prev_version = state.get("data_version", None)

    cur_version = get_data_version()

    if prev_version is None and cur_version is None:
        cycle_id = prev_cycle_id
    elif prev_version is None and cur_version is not None:
        cycle_id = prev_cycle_id
    elif prev_version is not None and cur_version is None:
        cycle_id = prev_cycle_id
    else:
        if str(cur_version) != str(prev_version):
            cycle_id = prev_cycle_id + 1
        else:
            cycle_id = prev_cycle_id

    state["cycle_id"] = int(cycle_id)
    state["data_version"] = cur_version
    state["updated_at"] = datetime.utcnow().isoformat()
    write_client_state(state)
    return int(cycle_id), cur_version


def push_status():
    ensure_bucket()

    cycle_id, data_version = get_or_bump_cycle_id()

    date = datetime.utcnow().strftime("%Y-%m-%d")
    key = f"status/{date}/{CLIENT_ID}.json"

    payload = {
        "client_id": CLIENT_ID,
        "cycle_id": cycle_id,
        "data_version": data_version,
        "has_data": HAS_DATA,
        "can_train": CAN_TRAIN,
        "timestamp": datetime.utcnow().isoformat(),
    }

    raw = json.dumps(payload).encode("utf-8")

    client.put_object(
        BUCKET,
        key,
        io.BytesIO(raw),
        length=len(raw),
        content_type="application/json",
    )

    print(
        f"[CLIENT-{CLIENT_ID}] Status uploaded "
        f"(cycle_id={cycle_id}, has_data={HAS_DATA}, can_train={CAN_TRAIN})"
    )


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    print(f"[CLIENT-{CLIENT_ID}] Starting status sender")
    push_status()
    print(f"[CLIENT-{CLIENT_ID}] Done")
