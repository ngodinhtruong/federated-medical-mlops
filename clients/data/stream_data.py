import json
import os
import numpy as np

CLUSTER_DIR = "/opt/fl/clusters"
STREAM_DIR = "/opt/fl/data_stream"


# ================= utils =================

def _get_client_id():
    cid = os.getenv("CLIENT_ID","A")
    if not cid:
        raise ValueError("CLIENT_ID env missing")
    return cid


def _cluster_path(cid):
    return os.path.join(CLUSTER_DIR, f"cluster_client_{cid}.npz")


def _stream_path(cid):
    os.makedirs(STREAM_DIR, exist_ok=True)
    return os.path.join(STREAM_DIR, f"stream_client_{cid}.npz")


def _pointer_path(cid):
    return os.path.join(STREAM_DIR, f"stream_pointer_{cid}.txt")


# ================= load pointer =================

def _load_pointer(path):
    if not os.path.exists(path):
        return {"train": 0, "eval": 0}

    with open(path, "r") as f:
        return json.load(f)


def _save_pointer(path, ptr):
    with open(path, "w") as f:
        json.dump(ptr, f)


# ================= main generator =================

def generate_batch(batch_size=50):

    cid = _get_client_id()

    cluster_file = _cluster_path(cid)
    stream_file = _stream_path(cid)
    pointer_file = _pointer_path(cid)

    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Cluster file missing: {cluster_file}")

    data = np.load(cluster_file)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_eval = data["X_eval"]
    y_eval = data["y_eval"]

    ptr = _load_pointer(pointer_file)

    train_ptr = int(ptr["train"])
    eval_ptr = int(ptr["eval"])

    total_train = len(X_train)
    total_eval = len(X_eval)

    # ===== train batch =====
    if train_ptr >= total_train:
        X_new_train = np.empty((0, *X_train.shape[1:]))
        y_new_train = np.empty((0, *y_train.shape[1:]))
    else:
        end = min(train_ptr + batch_size, total_train)
        X_new_train = X_train[train_ptr:end]
        y_new_train = y_train[train_ptr:end]
        train_ptr = end

    # ===== eval batch =====
    if eval_ptr >= total_eval:
        X_new_eval = np.empty((0, *X_eval.shape[1:]))
        y_new_eval = np.empty((0, *y_eval.shape[1:]))
    else:
        end = min(eval_ptr + batch_size, total_eval)
        X_new_eval = X_eval[eval_ptr:end]
        y_new_eval = y_eval[eval_ptr:end]
        eval_ptr = end

    # nếu cả train và eval đều rỗng → dataset exhausted
    if len(X_new_train) == 0 and len(X_new_eval) == 0:
        print(f"[STREAM] client={cid} dataset exhausted → no new data")
        return

    ptr = {
        "train": train_ptr,
        "eval": eval_ptr
        }

    _save_pointer(pointer_file, ptr)

    # ===== append stream =====
    if os.path.exists(stream_file):

        old = np.load(stream_file)

        Xs_train = np.concatenate([old["X_train"], X_new_train])
        ys_train = np.concatenate([old["y_train"], y_new_train])

        Xs_eval = np.concatenate([old["X_eval"], X_new_eval])
        ys_eval = np.concatenate([old["y_eval"], y_new_eval])

    else:

        Xs_train = X_new_train
        ys_train = y_new_train

        Xs_eval = X_new_eval
        ys_eval = y_new_eval

    np.savez(
        stream_file,
        X_train=Xs_train,
        y_train=ys_train,
        X_eval=Xs_eval,
        y_eval=ys_eval
    )

    print(
        f"[STREAM] client={cid} "
        f"train+={len(X_new_train)} "
        f"eval+={len(X_new_eval)} "
        f"train_ptr={train_ptr}/{total_train} "
        f"eval_ptr={eval_ptr}/{total_eval}"
    )

# ================= run =================

if __name__ == "__main__":

    batch_size = int(os.getenv("STREAM_BATCH", "50"))

    generate_batch(batch_size)