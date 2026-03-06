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

        # env
        self.client_id = os.getenv("CLIENT_ID", "A")
        self.total_clients = int(os.getenv("NUM_CLIENTS", "3"))
        self.client_ratios = os.getenv("CLIENT_RATIOS", None)

        self.val_ratio = float(os.getenv("VAL_RATIO", "0.2"))

    # ---------- utils ----------
    def _ds_to_numpy(self, ds):
        X, y = [], []
        for img, label in ds:
            X.append(img.numpy())
            y.append(label)
        return np.array(X, np.float32), np.array(y, np.int64)

    # ---------- stratified split ----------
    def _stratified_split(self, X_all, y_all):

        rng = np.random.default_rng(self.seed)

        if self.client_ratios:
            ratios = np.array([float(x) for x in self.client_ratios.split(",")])
            ratios = ratios / ratios.sum()
        else:
            ratios = np.ones(self.total_clients) / self.total_clients

        splits = [[] for _ in range(self.total_clients)]

        labels = np.unique(y_all)

        for label in labels:

            label_idx = np.where(y_all == label)[0]
            rng.shuffle(label_idx)

            n_label = len(label_idx)

            sizes = (ratios * n_label).astype(int)
            sizes[-1] = n_label - sizes[:-1].sum()

            start = 0
            for cid in range(self.total_clients):
                end = start + sizes[cid]
                splits[cid].extend(label_idx[start:end])
                start = end

        splits = [np.array(s) for s in splits]

        return splits

    # ---------- main ----------
    def build_for_env(self):

        out_path = f"{self.out_dir}/cluster_client_{self.client_id}.npz"

        if os.path.exists(out_path):
            print(f"[CLIENT {self.client_id}] Exists → skip")
            return

        print(f"[CLIENT {self.client_id}] Building dataset")

        train = PneumoniaMNIST("train", transform=self.transform, download=True)
        val = PneumoniaMNIST("val", transform=self.transform, download=True)

        Xtr, ytr = self._ds_to_numpy(train)
        Xv, yv = self._ds_to_numpy(val)

        X_all = np.concatenate([Xtr, Xv])
        y_all = np.concatenate([ytr, yv])

        splits = self._stratified_split(X_all, y_all)

        cid_list = [chr(ord("A") + i) for i in range(self.total_clients)]
        cid_to_idx = dict(zip(cid_list, splits))

        my_idx = cid_to_idx[self.client_id]

        X_client = X_all[my_idx]
        y_client = y_all[my_idx]

        # ---------- split train / eval ----------
        rng = np.random.default_rng(self.seed)

        perm = rng.permutation(len(X_client))

        n_val = int(len(X_client) * self.val_ratio)

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        X_train = X_client[train_idx]
        y_train = y_client[train_idx]

        X_eval = X_client[val_idx]
        y_eval = y_client[val_idx]

        np.savez_compressed(
            out_path,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval
        )

        print(f"[CLIENT {self.client_id}]")
        print("Train:", len(X_train))
        print("Eval :", len(X_eval))

    # ---------- loader ----------
    @staticmethod
    def load_my_cluster(out_dir="clusters"):

        cid = os.getenv("CLIENT_ID", "A")
        path = f"{out_dir}/cluster_client_{cid}.npz"

        data = np.load(path)

        return (
            data["X_train"],
            data["y_train"],
            data["X_eval"],
            data["y_eval"],
        )


if __name__ == "__main__":

    role = os.getenv("ROLE", "client").lower()
    if role != "client":
        raise RuntimeError("CLIENT only")

    builder = PneumoniaClusterBuilder(
        out_dir=os.getenv("OUT_DIR", "clusters"),
        seed=int(os.getenv("DATA_SEED", "42")),
    )

    builder.build_for_env()