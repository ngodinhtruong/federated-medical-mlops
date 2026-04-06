import numpy as np
import torch
import torch.nn as nn
import flwr as fl

from model.MLP import MLP
from model.SimpleCNN import SimpleCNN
from model.LogisticRegression import LogisticRegression
import os

def build_model_from_parameters(params, input_shape):
    flat_dim = int(np.prod(input_shape))
    prefix = os.getenv("MINIO_PREFIX", "training/mlp")
    
    if "cnn" in prefix.lower():
        channels = 1 if len(input_shape) == 2 else input_shape[0]
        model = SimpleCNN(input_channels=channels)
    elif "logreg" in prefix.lower():
        model = LogisticRegression(input_dim=flat_dim)
    else:
        model = MLP(input_dim=flat_dim)

    model.eval()

    weights = fl.common.parameters_to_ndarrays(params)

    state_dict = model.state_dict()

    for k, w in zip(state_dict.keys(), weights):

        state_dict[k] = torch.tensor(w)

    model.load_state_dict(state_dict)

    return model
def binary_metrics(y_true, y_pred):

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    eps = 1e-8

    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    return accuracy, precision, recall, f1





def server_val_loss(params, val_X, val_y):

    model = build_model_from_parameters(params, val_X.shape[1:])

    loss_fn = nn.BCELoss()

    prefix = os.getenv("MINIO_PREFIX", "training/mlp")
    with torch.no_grad():
        if "cnn" not in prefix.lower():
            X = val_X.view(val_X.size(0), -1)
        else:
            X = val_X

        pred = model(X)

        loss = loss_fn(pred, val_y.float())

    return float(loss.item())


def server_eval(params, val_X, val_y):

    model = build_model_from_parameters(params, val_X.shape[1:])

    prefix = os.getenv("MINIO_PREFIX", "training/mlp")
    with torch.no_grad():
        if "cnn" not in prefix.lower():
            X = val_X.view(val_X.size(0), -1)
        else:
            X = val_X

        logits = model(X)

        preds = (logits > 0.5).int().cpu().numpy()

    y_true = val_y.cpu().numpy()

    y_pred = preds

    acc, prec, rec, f1 = binary_metrics(y_true, y_pred)

    loss = server_val_loss(params, val_X, val_y)

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }