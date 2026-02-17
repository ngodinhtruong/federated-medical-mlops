# client.py
import os
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.MLP import MLP
from data.load_data import load_data_split

DEVICE = "cpu"

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

LOG_DIR = "/opt/fl/logs"
os.makedirs(LOG_DIR, exist_ok=True)

run_ts = datetime.now(VN_TZ).strftime("%Y-%m-%d_%H-%M-%S")
CLIENT_ID_FOR_LOG = os.getenv("CLIENT_ID", "unknown")
LOG_FILE = os.path.join(LOG_DIR, f"fl_client_{CLIENT_ID_FOR_LOG}_{run_ts}.log")

logger = logging.getLogger(f"fl_client_{CLIENT_ID_FOR_LOG}")
logger.setLevel(logging.INFO)


class VNFormatter(logging.Formatter):
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, VN_TZ).timetuple()


formatter = VNFormatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info(f"Logging initialized → {LOG_FILE}")


class FLClient(fl.client.NumPyClient):
    def __init__(self, seed):
        self.client_id = os.getenv("CLIENT_ID", "unknown")
        self.stop_after_round = int(os.getenv("CLIENT_STOP_AFTER_ROUND", "0"))

        self.delay_sec = 3
        self.delay_jitter_sec = 3

        self.train_set, self.val_set = load_data_split(seed=seed)

        X0, _ = self.train_set[0]
        input_dim = int(torch.numel(X0))
        self.model = MLP(input_dim=input_dim).to(DEVICE)

        self.train_loader = DataLoader(self.train_set, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=64, shuffle=False)

        self.loss_fn = nn.BCELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.01)

        logger.info(
            f"[CLIENT-{self.client_id}] Init done |"
            f"train={len(self.train_set)} val={len(self.val_set)} | "
            f"delay={self.delay_sec}s jitter={self.delay_jitter_sec}s"
        )

    def get_parameters(self, config):
        return [v.detach().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(
            zip(
                self.model.state_dict().keys(),
                [torch.tensor(p) for p in parameters],
            )
        )
        self.model.load_state_dict(state_dict)

    def train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for Xb, yb in self.train_loader:
            Xb = Xb.view(Xb.size(0), -1)
            pred = self.model(Xb)
            loss = self.loss_fn(pred, yb)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.item() * len(yb)
            correct += ((pred > 0.5) == yb).sum().item()
            total += len(yb)

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for Xb, yb in self.val_loader:
            Xb = Xb.view(Xb.size(0), -1)
            pred = self.model(Xb)
            loss = self.loss_fn(pred, yb)

            total_loss += loss.item() * len(yb)
            correct += ((pred > 0.5) == yb).sum().item()
            total += len(yb)

        return total_loss / total, correct / total

    def simulate_delay(self):
        if self.delay_sec <= 0 and self.delay_jitter_sec <= 0:
            return 0.0

        jitter = 0.0
        if self.delay_jitter_sec > 0:
            jitter = float(torch.rand(1).item()) * self.delay_jitter_sec

        total_delay = self.delay_sec + jitter
        logger.info(
            f"[CLIENT-{self.client_id}] Simulating delay: {total_delay:.3f}s "
            f"(base={self.delay_sec}s, jitter={jitter:.3f}s)"
        )
        time.sleep(total_delay)
        return total_delay

    def fit(self, parameters, config):
        round_id = int(config.get("server_round", 0))

        self.set_parameters(parameters)

        simulated = self.simulate_delay()
        epochs = 3
        for _ in range(epochs):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

        if self.stop_after_round > 0 and round_id >= self.stop_after_round:
            raise SystemExit(0)

        return (
            self.get_parameters(None),
            len(self.train_set),
            {
                "client_id": self.client_id,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "base_version": int(config.get("server_version", 0)),
                "simulated_delay_sec": float(simulated),
            },
        )

    def evaluate(self, parameters, config):
        round_id = config.get("server_round", "?")
        logger.info(f"[CLIENT-{self.client_id}] ===== GLOBAL EVAL ROUND {round_id} =====")

        self.set_parameters(parameters)
        val_loss, val_acc = self.validate()

        logger.info(
            f"[CLIENT-{self.client_id}][R{round_id}] "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        return val_loss, len(self.val_set), {"val_acc": val_acc}


if __name__ == "__main__":
    seed = int(os.environ.get("CLIENT_SEED", 0))
    server_address = os.environ.get("FL_SERVER_ADDRESS", "host.docker.internal:8080")

    logger.info(
        f"[CLIENT-{CLIENT_ID_FOR_LOG}] Starting client | seed={seed} | server={server_address}"
    )

    try:
        fl.client.start_client(
            server_address=server_address,
            client=FLClient(seed).to_client(),
        )
    except Exception:
        raise SystemExit(0)
