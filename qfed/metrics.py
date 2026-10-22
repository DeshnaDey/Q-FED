"""Evaluation metrics: classification scores and communication cost."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """`scores` are probabilities of the positive (malignant) class."""
    y_pred = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    logits = model(torch.as_tensor(X))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.as_tensor(y)).item()
    out = classification_metrics(y, torch.sigmoid(logits).numpy())
    out["loss"] = float(loss)
    return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def round_comm_bytes(n_params: int, n_clients: int, bytes_per_param: int = 4) -> int:
    """Parameters sent down to and back up from each participating client in one round."""
    return n_params * bytes_per_param * 2 * n_clients
