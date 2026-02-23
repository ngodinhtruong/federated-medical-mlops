import os
import io
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from minio import Minio

from model.MLP import MLP
from data.load_data import load_test_ds


def build_minio_client():
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "admin12345")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def minio_get_bytes(client: Minio, bucket: str, object_name: str) -> bytes:
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


def minio_put_json(client: Minio, bucket: str, object_name: str, payload: dict):
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    bio = io.BytesIO(raw)
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=bio,
        length=len(raw),
        content_type="application/json",
    )


def load_ndarrays_from_npz_bytes(raw: bytes):
    bio = io.BytesIO(raw)
    npz = np.load(bio, allow_pickle=False)
    keys = sorted(npz.files, key=lambda x: int(x.replace("arr_", "")) if x.startswith("arr_") else x)
    return [npz[k] for k in keys]


def build_model_from_ndarrays(ndarrays, input_shape):
    flat_dim = int(np.prod(input_shape))
    model = MLP(input_dim=flat_dim)
    model.eval()

    state_dict = model.state_dict()
    if len(state_dict.keys()) != len(ndarrays):
        raise ValueError(f"Mismatch weights: state_dict={len(state_dict.keys())} npz={len(ndarrays)}")

    for k, w in zip(state_dict.keys(), ndarrays):
        state_dict[k] = torch.tensor(w)

    model.load_state_dict(state_dict)
    return model


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    eps = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def eval_npz_weights(npz_raw: bytes, val_x: torch.Tensor, val_y: torch.Tensor, threshold: float):
    ndarrays = load_ndarrays_from_npz_bytes(npz_raw)
    model = build_model_from_ndarrays(ndarrays, val_x.shape[1:])
    loss_fn = nn.BCELoss()

    with torch.no_grad():
        x = val_x.view(val_x.size(0), -1)
        probs = model(x)
        loss = loss_fn(probs, val_y.float()).item()
        preds = (probs > threshold).int().cpu().numpy()

    y_true = val_y.int().cpu().numpy()
    metrics = binary_metrics(y_true, preds)
    metrics["loss"] = float(loss)
    metrics["threshold"] = float(threshold)
    return metrics


def choose_by_f1_then_loss(best_eval: dict, last_eval: dict):
    if best_eval["f1"] > last_eval["f1"]:
        return "best"
    if best_eval["f1"] < last_eval["f1"]:
        return "last"
    if best_eval["loss"] < last_eval["loss"]:
        return "best"
    if best_eval["loss"] > last_eval["loss"]:
        return "last"
    return "last"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_id", required=True)
    parser.add_argument("--bucket", default=os.getenv("MINIO_BUCKET", "fl-artifacts"))
    parser.add_argument("--prefix", default=os.getenv("MINIO_PREFIX", "cycles"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--update_champion", action="store_true", default=True)
    args = parser.parse_args()

    client = build_minio_client()

    prefix = args.prefix.rstrip("/")
    cycle_path = f"{prefix}/cycle_{args.cycle_id}"

    obj_best = f"{cycle_path}/model_best.npz"
    obj_last = f"{cycle_path}/model_last.npz"
    obj_meta = f"{cycle_path}/meta.json"
    obj_summary = f"{cycle_path}/summary.json"

    best_raw = minio_get_bytes(client, args.bucket, obj_best)
    last_raw = minio_get_bytes(client, args.bucket, obj_last)

    try:
        meta = json.loads(minio_get_bytes(client, args.bucket, obj_meta).decode("utf-8"))
    except Exception:
        meta = {"cycle_id": str(args.cycle_id)}

    try:
        summary = json.loads(minio_get_bytes(client, args.bucket, obj_summary).decode("utf-8"))
    except Exception:
        summary = {}

    val_x, val_y = load_test_ds()
    val_x = val_x.float()
    val_y = val_y.int()

    eval_best = eval_npz_weights(best_raw, val_x, val_y, args.threshold)
    eval_last = eval_npz_weights(last_raw, val_x, val_y, args.threshold)

    chosen = choose_by_f1_then_loss(eval_best, eval_last)
    chosen_obj = obj_best if chosen == "best" else obj_last

    eval_best_payload = {
        "cycle_id": str(args.cycle_id),
        "model": "best",
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": eval_best,
    }
    eval_last_payload = {
        "cycle_id": str(args.cycle_id),
        "model": "last",
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": eval_last,
    }

    minio_put_json(client, args.bucket, f"{cycle_path}/EVAL/eval_best.json", eval_best_payload)
    minio_put_json(client, args.bucket, f"{cycle_path}/EVAL/eval_last.json", eval_last_payload)

    chosen_payload = {
        "cycle_id": str(args.cycle_id),
        "criteria": "f1_then_loss",
        "chosen": chosen,
        "chosen_object": chosen_obj,
        "threshold": float(args.threshold),
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "best": eval_best,
        "last": eval_last,
        "meta": meta,
        "summary": summary,
    }
    minio_put_json(client, args.bucket, f"{cycle_path}/CHOSEN.json", chosen_payload)

    if args.update_champion:
        champion_payload = {
            "cycle_id": str(args.cycle_id),
            "chosen": chosen,
            "model_object": chosen_obj,
            "threshold": float(args.threshold),
            "metrics": eval_best if chosen == "best" else eval_last,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        minio_put_json(client, args.bucket, "registry/champion.json", champion_payload)


if __name__ == "__main__":
    main()