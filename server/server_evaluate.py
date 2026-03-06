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

    parser.add_argument(
        "--bucket",
        default=os.getenv("MINIO_BUCKET", "fl-artifacts"),
    )

    parser.add_argument(
        "--prefix",
        default=os.getenv("MINIO_PREFIX", "cycles"),
    )

    parser.add_argument(
        "--update_champion",
        action="store_true",
        default=True,
    )

    args = parser.parse_args()

    client = build_minio_client()

    prefix = args.prefix.rstrip("/")

    cycle_path = f"{prefix}/cycle_{args.cycle_id}"

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

    val_x, val_y = load_test_ds()

    val_x = val_x.float()
    val_y = val_y.int()

    eval_best = eval_pth_weights(best_raw, val_x, val_y)
    eval_last = eval_pth_weights(last_raw, val_x, val_y)

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

    minio_put_json(
        client,
        args.bucket,
        f"{cycle_path}/EVAL/eval_best.json",
        eval_best_payload,
    )

    minio_put_json(
        client,
        args.bucket,
        f"{cycle_path}/EVAL/eval_last.json",
        eval_last_payload,
    )

    chosen_payload = {
        "cycle_id": str(args.cycle_id),
        "criteria": "f1_then_loss",
        "chosen": chosen,
        "chosen_object": chosen_obj,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "best": eval_best,
        "last": eval_last,
        "meta": meta,
        "summary": summary,
    }

    minio_put_json(
        client,
        args.bucket,
        f"{cycle_path}/CHOSEN.json",
        chosen_payload,
    )

    if args.update_champion:

        champion_payload = {
            "cycle_id": str(args.cycle_id),
            "chosen": chosen,
            "model_object": chosen_obj,
            "metrics": eval_best if chosen == "best" else eval_last,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        minio_put_json(
            client,
            args.bucket,
            "registry/champion.json",
            champion_payload,
        )


if __name__ == "__main__":
    main()