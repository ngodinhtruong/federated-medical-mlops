import os
import numpy as np
from torchvision import transforms
from medmnist.dataset import PneumoniaMNIST


class PneumoniaClusterBuilder:

    def __init__(self, normalize=True, out_dir="clusters", seed=42):
        tf = [transforms.ToTensor()]
        if normalize:
            tf.append(transforms.Normalize(mean=[0.5], std=[0.5]))

        self.transform = transforms.Compose(tf)
        self.out_dir = out_dir
        self.seed = seed
        os.makedirs(out_dir, exist_ok=True)

        # CLIENT-only env
        self.client_id = os.getenv("CLIENT_ID", "A")
        self.total_clients = int(os.getenv("NUM_CLIENTS", "3"))

    # ---------- utils ----------
    def _ds_to_numpy(self, ds):
        X, y = [], []
        for img, label in ds:
            X.append(img.numpy())
            y.append(label)
        return np.array(X, np.float32), np.array(y, np.int64)

    # ---------- main ----------
    def build_for_env(self):

        out_path = f"{self.out_dir}/cluster_client_{self.client_id}.npz"

        # ✅ nếu đã có file → KHÔNG tải data
        if os.path.exists(out_path):
            print(f"[CLIENT {self.client_id}] Exists → skip build")
            return

        print(f"[CLIENT {self.client_id}] Building cluster")
        print(f"  NUM_CLIENTS = {self.total_clients}")
        print(f"  SEED        = {self.seed}")

        # -------- load ONLY train + val --------
        train = PneumoniaMNIST("train", transform=self.transform, download=True)
        val   = PneumoniaMNIST("val",   transform=self.transform, download=True)

        Xtr, ytr = self._ds_to_numpy(train)
        Xv,  yv  = self._ds_to_numpy(val)

        # -------- merge global pool --------
        X_all = np.concatenate([Xtr, Xv], axis=0)
        y_all = np.concatenate([ytr, yv], axis=0)

        n_total = len(X_all)

        # -------- deterministic split (NO LEAK) --------
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_total)

        splits = np.array_split(perm, self.total_clients)

        cid_list = [chr(ord("A") + i) for i in range(self.total_clients)]
        cid_to_idx = dict(zip(cid_list, splits))

        if self.client_id not in cid_to_idx:
            raise ValueError(f"CLIENT_ID {self.client_id} invalid")

        my_idx = cid_to_idx[self.client_id]

        # -------- save ONLY my data --------
        np.savez_compressed(
            out_path,
            X=X_all[my_idx],
            y=y_all[my_idx],
        )

        print(f"[CLIENT {self.client_id}] Saved {len(my_idx)} samples")
        print("[CLIENT] Done — no test, no leakage")

    # ---------- loader ----------
    @staticmethod
    def load_my_cluster(out_dir="clusters"):
        cid = os.getenv("CLIENT_ID", "A")
        path = f"{out_dir}/cluster_client_{cid}.npz"

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing — run build_for_env() first")

        data = np.load(path)
        return data["X"], data["y"]


if __name__ == "__main__":

    role = os.getenv("ROLE", "client").lower()
    if role != "client":
        raise RuntimeError("This script is CLIENT-only")

    print("=== CLIENT DATA BUILD ===")
    print("CLIENT_ID  :", os.getenv("CLIENT_ID"))
    print("NUM_CLIENTS:", os.getenv("NUM_CLIENTS"))
    print("SEED       :", os.getenv("DATA_SEED"))

    builder = PneumoniaClusterBuilder(
        out_dir=os.getenv("OUT_DIR", "clusters"),
        seed=int(os.getenv("DATA_SEED", "42")),
    )

    builder.build_for_env()
