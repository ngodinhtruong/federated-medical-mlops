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


def push_status():
    ensure_bucket()

    date = datetime.utcnow().strftime("%Y-%m-%d")
    key = f"status/{date}/{CLIENT_ID}.json"

    payload = {
        "client_id": CLIENT_ID,
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
        f"(has_data={HAS_DATA}, can_train={CAN_TRAIN})"
    )


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    print(f"[CLIENT-{CLIENT_ID}] Starting status sender")
    push_status()
    print(f"[CLIENT-{CLIENT_ID}] Done")
