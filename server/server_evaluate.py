import os
import io
import json
import argparse
from datetime import datetime

import torch
from minio import Minio

from flwr.common import ndarrays_to_parameters

from data.load_data import load_test_ds
from utils.model_utils import server_eval
from utils.minio_utils import minio_get_bytes, minio_put_json
from utils.minio_utils import load_fl_params_from_pth_bytes


def build_minio_client():
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "admin12345")
    secure = False
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def eval_pth_weights(raw: bytes, val_x: torch.Tensor, val_y: torch.Tensor):
    params = load_fl_params_from_pth_bytes(raw)
    metrics = server_eval(params, val_x, val_y)
    return metrics


def choose_by_f1_then_loss(eval_a: dict, eval_b: dict):
    """So sánh 2 bộ metrics. Trả về 'a' nếu a tốt hơn, 'b' nếu b tốt hơn."""
    if eval_a["f1"] > eval_b["f1"]:
        return "a"
    if eval_a["f1"] < eval_b["f1"]:
        return "b"
    # F1 bằng nhau → so loss (thấp hơn = tốt hơn)
    if eval_a["loss"] < eval_b["loss"]:
        return "a"
    if eval_a["loss"] > eval_b["loss"]:
        return "b"
    # Hoàn toàn bằng nhau → giữ nguyên (ưu tiên a = incumbent)
    return "a"


def prefix_to_model_type(prefix: str) -> str:
    """Suy ra tên model từ prefix. Ví dụ: training/cnn → CNN"""
    p = prefix.lower()
    if "cnn" in p:
        return "CNN"
    elif "logreg" in p:
        return "LogisticRegression"
    else:
        return "MLP"


def read_current_champion(client, bucket: str) -> dict | None:
    """Đọc champion hiện tại từ MinIO. Trả về None nếu chưa có."""
    try:
        raw = minio_get_bytes(client, bucket, "registry/champion.json")
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle_id", required=True)
    parser.add_argument(
        "--bucket",
        default=os.getenv("MINIO_BUCKET", "fl-artifacts"),
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("MINIO_PREFIX", "training/mlp"),
    )
    args = parser.parse_args()

    client = build_minio_client()
    prefix = args.prefix.rstrip("/")
    model_type = prefix_to_model_type(prefix)
    cycle_path = f"{prefix}/cycle_{args.cycle_id}"

    print(f"[EVALUATE] model_type={model_type} cycle={cycle_path}")

    # ===== 1. Đọc model weights từ MinIO =====
    obj_best = f"{cycle_path}/model_best.pth"
    obj_last = f"{cycle_path}/model_last.pth"
    obj_meta = f"{cycle_path}/meta.json"
    obj_summary = f"{cycle_path}/summary.json"

    best_raw = minio_get_bytes(client, args.bucket, obj_best)
    last_raw = minio_get_bytes(client, args.bucket, obj_last)

    try:
        meta = json.loads(
            minio_get_bytes(client, args.bucket, obj_meta).decode("utf-8")
        )
    except Exception:
        meta = {"cycle_id": str(args.cycle_id)}

    try:
        summary = json.loads(
            minio_get_bytes(client, args.bucket, obj_summary).decode("utf-8")
        )
    except Exception:
        summary = {}

    # ===== 2. Evaluate trên test set =====
    val_x, val_y = load_test_ds()
    val_x = val_x.float()
    val_y = val_y.int()

    eval_best = eval_pth_weights(best_raw, val_x, val_y)
    eval_last = eval_pth_weights(last_raw, val_x, val_y)

    print(f"[EVALUATE] eval_best: f1={eval_best['f1']:.4f} loss={eval_best['loss']:.4f}")
    print(f"[EVALUATE] eval_last: f1={eval_last['f1']:.4f} loss={eval_last['loss']:.4f}")

    # ===== 3. Chọn winner trong cycle này (best vs last) =====
    cycle_winner = choose_by_f1_then_loss(eval_best, eval_last)
    if cycle_winner == "a":
        winner_label = "best"
        winner_metrics = eval_best
        winner_obj = obj_best
    else:
        winner_label = "last"
        winner_metrics = eval_last
        winner_obj = obj_last

    print(f"[EVALUATE] Cycle winner: {winner_label} (f1={winner_metrics['f1']:.4f})")

    # ===== 4. Lưu kết quả evaluate vào cycle path =====
    minio_put_json(client, args.bucket, f"{cycle_path}/EVAL/eval_best.json", {
        "cycle_id": str(args.cycle_id),
        "model_type": model_type,
        "checkpoint": "best",
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": eval_best,
    })

    minio_put_json(client, args.bucket, f"{cycle_path}/EVAL/eval_last.json", {
        "cycle_id": str(args.cycle_id),
        "model_type": model_type,
        "checkpoint": "last",
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": eval_last,
    })

    minio_put_json(client, args.bucket, f"{cycle_path}/CHOSEN.json", {
        "cycle_id": str(args.cycle_id),
        "model_type": model_type,
        "criteria": "f1_then_loss",
        "chosen": winner_label,
        "chosen_object": winner_obj,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "best": eval_best,
        "last": eval_last,
        "meta": meta,
        "summary": summary,
    })

    # ===== 5. Champion Challenge — so sánh với champion hiện tại =====
    current_champion = read_current_champion(client, args.bucket)

    challenge_result = {
        "challenger_model_type": model_type,
        "challenger_prefix": prefix,
        "challenger_cycle_id": str(args.cycle_id),
        "challenger_metrics": winner_metrics,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    promoted = False

    if current_champion is None:
        # Chưa có champion → tự động thăng hạng
        promoted = True
        challenge_result["reason"] = "no_existing_champion"
        print(f"[CHAMPION] No existing champion → promoting {model_type}")

    else:
        champion_metrics = current_champion.get("metrics", {})
        champion_model = current_champion.get("model_type", "unknown")

        challenge_result["champion_model_type"] = champion_model
        challenge_result["champion_metrics"] = champion_metrics

        compare = choose_by_f1_then_loss(winner_metrics, champion_metrics)

        if compare == "a":
            promoted = True
            challenge_result["reason"] = "challenger_outperforms_champion"
            print(
                f"[CHAMPION] 🏆 {model_type} (f1={winner_metrics['f1']:.4f}) "
                f"beats {champion_model} (f1={champion_metrics.get('f1', 0):.4f}) → NEW CHAMPION!"
            )
        else:
            challenge_result["reason"] = "champion_retains"
            print(
                f"[CHAMPION] ❌ {model_type} (f1={winner_metrics['f1']:.4f}) "
                f"does not beat {champion_model} (f1={champion_metrics.get('f1', 0):.4f}) → champion retained"
            )

    challenge_result["promoted"] = promoted

    # Lưu lịch sử challenge
    minio_put_json(
        client, args.bucket,
        f"registry/challenges/{model_type}_{args.cycle_id}.json",
        challenge_result,
    )

    # Cập nhật champion nếu thắng
    if promoted:
        champion_payload = {
            "model_type": model_type,
            "prefix": prefix,
            "cycle_id": str(args.cycle_id),
            "chosen": winner_label,
            "model_object": winner_obj,
            "metrics": winner_metrics,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        minio_put_json(client, args.bucket, "registry/champion.json", champion_payload)
        print(f"[CHAMPION] registry/champion.json updated → {model_type}")


if __name__ == "__main__":
    main()