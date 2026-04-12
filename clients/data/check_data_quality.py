"""
Evidently AI — Data Drift Check (Lightweight)
Chỉ check DataDrift trên label + 10 features mẫu → nhanh gấp 10x.
"""
import os
import io
import json
import hashlib
import numpy as np
from datetime import datetime

print("[DATA-QUALITY] Basic imports OK", flush=True)

# ==================== CONFIG ====================

CLIENT_ID = os.getenv("CLIENT_ID", "A")
STREAM_DIR = "/opt/fl/data_stream"
SNAPSHOT_DIR = "/opt/fl/data_stream"

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin12345")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET = os.getenv("MINIO_BUCKET", "fl-artifacts")

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
MAX_FEATURES = 10  # Chỉ check 10 features mẫu thay vì toàn bộ 700+

# ==================== HELPERS ====================

def _minio_client():
    from minio import Minio
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)


def _ensure_bucket(mc):
    if not mc.bucket_exists(BUCKET):
        mc.make_bucket(BUCKET)


def _stream_path():
    return os.path.join(STREAM_DIR, f"stream_client_{CLIENT_ID}.npz")


def _snapshot_path():
    return os.path.join(SNAPSHOT_DIR, f"stream_client_{CLIENT_ID}_prev_snapshot.npz")


def _npz_to_dataframe(npz_path, max_features=MAX_FEATURES):
    """Chuyển npz sang DataFrame, chỉ lấy label + max_features cột."""
    import pandas as pd
    data = np.load(npz_path)
    X = data["X_train"]
    y = data["y_train"]

    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    # Chỉ lấy max_features cột (đều nhau) thay vì toàn bộ
    total_cols = X_flat.shape[1]
    if total_cols > max_features:
        indices = np.linspace(0, total_cols - 1, max_features, dtype=int)
        X_flat = X_flat[:, indices]

    cols = [f"f{i}" for i in range(X_flat.shape[1])]
    df = pd.DataFrame(X_flat, columns=cols)
    df["label"] = y.flatten()
    return df


def _compute_hash(npz_path):
    h = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ==================== MAIN ====================

def run_check():
    stream_file = _stream_path()
    snapshot_file = _snapshot_path()

    if not os.path.exists(stream_file):
        print(f"[DATA-QUALITY] Client={CLIENT_ID}: No stream data yet, skipping.", flush=True)
        return

    current_hash = _compute_hash(stream_file)
    current_df = _npz_to_dataframe(stream_file)
    print(f"[DATA-QUALITY] Client={CLIENT_ID}: Current data shape={current_df.shape}", flush=True)

    has_reference = os.path.exists(snapshot_file)

    if not has_reference:
        print(f"[DATA-QUALITY] Client={CLIENT_ID}: First run — saving baseline.", flush=True)
        data = np.load(stream_file)
        np.savez(snapshot_file, X_train=data["X_train"], y_train=data["y_train"],
                 X_eval=data["X_eval"], y_eval=data["y_eval"])
        print(f"[DATA-QUALITY] Client={CLIENT_ID}: Baseline saved. Skipping check.", flush=True)
        return

    reference_df = _npz_to_dataframe(snapshot_file)
    print(f"[DATA-QUALITY] Client={CLIENT_ID}: Reference shape={reference_df.shape}", flush=True)

    # ===== Chạy Evidently — chỉ DataDrift, không DataQuality =====
    print(f"[DATA-QUALITY] Importing evidently...", flush=True)
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    print(f"[DATA-QUALITY] Evidently imported OK", flush=True)

    report = Report(metrics=[DataDriftPreset()])
    print(f"[DATA-QUALITY] Running drift check...", flush=True)
    report.run(reference_data=reference_df, current_data=current_df)
    print(f"[DATA-QUALITY] Drift check done!", flush=True)

    # ===== Kết quả =====
    report_dict = report.as_dict()
    drift_detected = False
    drift_share = 0.0
    drift_alert = False

    try:
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            if "share_of_drifted_columns" in result:
                drift_share = result["share_of_drifted_columns"]
                drift_detected = result.get("dataset_drift", False)
                break
    except Exception as e:
        print(f"[DATA-QUALITY] Warning: parse error: {e}", flush=True)

    if drift_share >= DRIFT_THRESHOLD:
        drift_alert = True
        print(f"[DATA-QUALITY] ⚠️ DRIFT ALERT! drift={drift_share:.0%}", flush=True)
    elif drift_detected:
        print(f"[DATA-QUALITY] ⚡ Mild drift: {drift_share:.0%}", flush=True)
    else:
        print(f"[DATA-QUALITY] ✅ No drift: {drift_share:.0%}", flush=True)

    # ===== Upload JSON lên MinIO =====
    mc = _minio_client()
    _ensure_bucket(mc)

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    ts_str = datetime.utcnow().strftime("%H-%M-%S")
    base_key = f"clients/data-quality/{CLIENT_ID}/{date_str}"

    summary = {
        "client_id": CLIENT_ID,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_hash": current_hash,
        "current_samples": int(current_df.shape[0]),
        "reference_samples": int(reference_df.shape[0]),
        "features_checked": int(current_df.shape[1] - 1),
        "drift_detected": drift_detected,
        "drift_share": round(drift_share, 4),
        "drift_alert": drift_alert,
        "drift_threshold": DRIFT_THRESHOLD,
    }

    raw = json.dumps(summary, indent=2).encode("utf-8")
    mc.put_object(BUCKET, f"{base_key}/report_{ts_str}.json",
                  io.BytesIO(raw), length=len(raw), content_type="application/json")

    print(f"[DATA-QUALITY] Client={CLIENT_ID}: Report uploaded → {base_key}/", flush=True)

    # ===== Cập nhật snapshot =====
    data = np.load(stream_file)
    np.savez(snapshot_file, X_train=data["X_train"], y_train=data["y_train"],
             X_eval=data["X_eval"], y_eval=data["y_eval"])
    print(f"[DATA-QUALITY] Client={CLIENT_ID}: Snapshot updated.", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        print(f"[DATA-QUALITY] ========== START ==========", flush=True)
        print(f"[DATA-QUALITY] Client={CLIENT_ID} MINIO={MINIO_ENDPOINT}", flush=True)
        run_check()
        print(f"[DATA-QUALITY] ========== DONE ==========", flush=True)
    except Exception as e:
        print(f"[DATA-QUALITY] ❌ ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        print(f"[DATA-QUALITY] ⚠️ Continuing (non-blocking)", flush=True)
