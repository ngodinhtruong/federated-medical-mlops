import os
import numpy as np
import torch
from torch.utils.data import TensorDataset


CLUSTER_DIR = "clusters"


# ================= path resolve =================

def _get_cluster_path():

    role = os.getenv("ROLE", "").lower()
    client_id = os.getenv("CLIENT_ID")

    if role == "server":
        name = "cluster_server.npz"

    elif role == "client":
        if not client_id:
            raise ValueError("CLIENT_ID env is required when ROLE=client")
        name = f"cluster_client_{client_id}.npz"

    else:
        raise ValueError("ROLE must be set to 'client' or 'server'")

    path = os.path.join(CLUSTER_DIR, name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[DATA] Cluster file not found: {path}\n"
            f"→ Build cluster for this ENV first"
        )

    return path


# ================= tensor convert =================

def _to_tensor(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    if y.ndim == 1:
        y = y.unsqueeze(1)

    return X, y


# ================= main loader =================

def load_data_split(seed=0, n_default=1000, val_ratio_default=0.2):

    # ----- ENV controls -----
    n = int(os.getenv("DATA_N", n_default))
    val_ratio = float(os.getenv("VAL_RATIO", val_ratio_default))
    seed = int(os.getenv("DATA_SEED", seed))

    path = _get_cluster_path()
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
        f"[DATA] ROLE={os.getenv('ROLE')} "
        f"CLIENT={os.getenv('CLIENT_ID')} "
        f"cluster={os.path.basename(path)} "
        f"subset={n} train={len(train_ds)} val={len(val_ds)}"
    )

    return train_ds, val_ds



def load_test_ds():

    role = os.getenv("ROLE", "server").lower()
    if role != "server":
        raise PermissionError("Only ROLE=server can load global test set")

    path = os.path.join(CLUSTER_DIR, "cluster_server.npz")

    if not os.path.exists(path):
        raise FileNotFoundError("cluster_server.npz not found")

    data = np.load(path)

    if "X_test" not in data:
        raise ValueError("server cluster missing X_test")

    X, y = _to_tensor(data["X_test"], data["y_test"])

    print(f"[DATA] SERVER global test={len(X)}")

    return X, y
