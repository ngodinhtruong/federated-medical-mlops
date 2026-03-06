# load stream data
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset


STREAM_DIR = "data_stream"


# ================= path resolve (CLIENT ONLY) =================

def _get_client_stream_path():

    role = os.getenv("ROLE", "client").lower()
    if role != "client":
        raise RuntimeError("This loader is CLIENT-only")

    client_id = os.getenv("CLIENT_ID",'A')
    if not client_id:
        raise ValueError("CLIENT_ID env is required")

    name = f"stream_client_{client_id}.npz"
    path = os.path.join(STREAM_DIR, name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[DATA] Stream dataset not found: {path}\n"
            f"→ Wait for Airflow ingest_new_data to generate stream first"
        )

    return path


# ================= tensor convert =================

def _to_tensor(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    if y.ndim == 1:
        y = y.unsqueeze(1)

    return X, y


def load_data_split(client_id,seed):


    seed_env = os.getenv("DATA_SEED",'1')
    if seed_env:
        seed = int(seed_env)
    path = _get_client_stream_path()
    data = np.load(path)

    if "X_train" not in data:
        raise ValueError(f"{path} is not a valid stream dataset")

    X_train, y_train = _to_tensor(data["X_train"], data["y_train"])
    X_eval, y_eval = _to_tensor(data["X_eval"], data["y_eval"])



    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_eval, y_eval)

    print(
        f"[DATA][CLIENT] id={os.getenv('CLIENT_ID','A')} "
        f"stream={os.path.basename(path)} "
        f"train={len(train_ds)} val={len(val_ds)}"
    )

    return train_ds, val_ds



load_data_split('A','1')