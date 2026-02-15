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

    # ---------- utils ----------
    def _ds_to_numpy(self, ds):
        X, y = [], []
        for img, label in ds:
            X.append(img.numpy())
            y.append(label)
        return np.array(X, np.float32), np.array(y, np.int64)

    # ---------- main ----------
    def build_server_only(self):

        out_path = f"{self.out_dir}/cluster_server.npz"

        if os.path.exists(out_path):
            print(f"[DATA] Exists → skip build: {out_path}")
            return

        print("[DATA] Building SERVER cluster (VAL + TEST only)")

        # -------- load dataset (ONLY val + test) --------
        val  = PneumoniaMNIST("val",  transform=self.transform, download=True)
        test = PneumoniaMNIST("test", transform=self.transform, download=True)

        X_val, y_val = self._ds_to_numpy(val)
        X_test, y_test = self._ds_to_numpy(test)

        # -------- save server cluster --------
        np.savez_compressed(
            out_path,
            X=X_val,
            y=y_val,
            X_test=X_test,
            y_test=y_test
        )

        print(f"[DATA] Saved SERVER cluster")
        print(f"       VAL  : {len(X_val)} samples")
        print(f"       TEST : {len(X_test)} samples")
        print("[DATA] Done")

    # ---------- loader ----------
    @staticmethod
    def load_server_train(out_dir="clusters"):
        data = np.load(f"{out_dir}/cluster_server.npz")
        return data["X"], data["y"]

    @staticmethod
    def load_server_test(out_dir="clusters"):
        data = np.load(f"{out_dir}/cluster_server.npz")
        return data["X_test"], data["y_test"]


if __name__ == "__main__":

    seed = int(os.getenv("DATA_SEED", "42"))

    builder = PneumoniaClusterBuilder(
        out_dir="clusters",
        seed=seed
    )

    builder.build_server_only()
