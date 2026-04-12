import io
import json
import torch
import numpy as np
from datetime import datetime

from flwr.common import parameters_to_ndarrays

from flwr.common import ndarrays_to_parameters

def minio_put_json(minio_client, bucket, object_name, payload):
    raw = json.dumps(payload).encode("utf-8")
    bio = io.BytesIO(raw)

    minio_client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=bio,
        length=len(raw),
        content_type="application/json",
    )


def minio_put_bytes(minio_client, bucket, object_name, raw, content_type):
    bio = io.BytesIO(raw)

    minio_client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=bio,
        length=len(raw),
        content_type=content_type,
    )
def minio_get_bytes(minio_client, bucket, object_name): 
    resp = minio_client.get_object(bucket, object_name)
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


def fl_params_to_pth_bytes(params):
    nds = parameters_to_ndarrays(params)

    state_dict = {}

    for i, w in enumerate(nds):
        state_dict[f"param_{i}"] = torch.tensor(w)

    buf = io.BytesIO()

    torch.save(state_dict, buf)

    buf.seek(0)

    return buf.getvalue()
def load_fl_params_from_pth_bytes(raw):

    buf = io.BytesIO(raw)

    state_dict = torch.load(buf, map_location="cpu")

    nds = []

    for k in sorted(state_dict.keys(), key=lambda x: int(x.split("_")[1])):
        nds.append(state_dict[k].cpu().numpy())

    return ndarrays_to_parameters(nds)

def get_eligible_clients(minio_client, bucket):

    date = datetime.utcnow().strftime("%Y-%m-%d")

    prefix = f"clients/status/{date}/"

    eligible = []

    for obj in minio_client.list_objects(bucket, prefix=prefix, recursive=True):

        raw = minio_client.get_object(bucket, obj.object_name).read()

        data = json.loads(raw)

        if data.get("has_data") and data.get("can_train"):

            eligible.append(str(data.get("client_id")))

    return eligible