import os
import numpy as np
import torch
from torch.utils.data import TensorDataset


CLUSTER_DIR = "clusters"


# ================= path resolve (CLIENT ONLY) =================

def _get_client_cluster_path():

    role = os.getenv("ROLE", "client").lower()
    if role != "client":
        raise RuntimeError("This loader is CLIENT-only")

    client_id = os.getenv("CLIENT_ID")
    if not client_id:
        raise ValueError("CLIENT_ID env is required")

    name = f"cluster_client_{client_id}.npz"
    path = os.path.join(CLUSTER_DIR, name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[DATA] Client cluster file not found: {path}\n"
            f"→ Build client cluster first"
        )

    return path


# ================= tensor convert =================

def _to_tensor(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    if y.ndim == 1:
        y = y.unsqueeze(1)

    return X, y



def load_data_split(seed=0, n_default=1000, val_ratio_default=0.2):

    # ----- ENV controls -----
    n = int(os.getenv("DATA_N", n_default))
    val_ratio = float(os.getenv("VAL_RATIO", val_ratio_default))

    seed_env = os.getenv("DATA_SEED")
    if seed_env:
        seed = int(seed_env)

    path = _get_client_cluster_path()
    data = np.load(path)

    if "X" not in data:
        raise ValueError(f"{path} is not a client cluster file")

    X, y = _to_tensor(data["X"], data["y"])

    total_n = len(X)
    n = min(n, total_n)

    # ----- deterministic subset -----
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total_n, generator=g)[:n]

    X = X[idx]
    y = y[idx]

    # ----- train/val split -----
    g2 = torch.Generator().manual_seed(seed + 999)
    perm = torch.randperm(n, generator=g2)

    n_val = int(n * val_ratio)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds   = TensorDataset(X[val_idx],   y[val_idx])

    print(
        f"[DATA][CLIENT] id={os.getenv('CLIENT_ID')} "
        f"cluster={os.path.basename(path)} "
        f"subset={n} train={len(train_ds)} val={len(val_ds)}"
    )

    return train_ds, val_ds
