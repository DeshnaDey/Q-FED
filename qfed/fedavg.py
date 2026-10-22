"""Federated Averaging, written out step by step (slide 11 of the presentation).

    1 INITIALISE   server creates the global parameters
    2 DISTRIBUTE   global parameters are copied to each sampled client
    3 TRAIN        each client trains on its own data only
    4 SEND         clients return parameter vectors, never raw records
    5 AGGREGATE    theta_global = sum_k (n_k / n) * theta_k
    6 REDISTRIBUTE updated global parameters go back out
    7 REPEAT       steps 2-6 for `rounds` communication rounds
    8 EVALUATE     the global model is tested on the shared held-out test set
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
from torch import nn

from qfed.metrics import count_params, evaluate, round_comm_bytes

Client = tuple[np.ndarray, np.ndarray]
RoundCallback = Callable[[dict], None]


def fedavg(states: list[dict], weights: list[float]) -> OrderedDict:
    """Weighted average of state dicts. All clients must share one architecture and key order."""
    total = float(sum(weights))
    keys = states[0].keys()
    for s in states[1:]:
        if s.keys() != keys:
            raise ValueError("Client models have different parameter layouts; cannot average")
    return OrderedDict(
        (k, sum(s[k].detach().double() * (w / total) for s, w in zip(states, weights)).to(states[0][k].dtype))
        for k in keys
    )


def train_local(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
                lr: float, generator: torch.Generator) -> tuple[float, int]:
    """Train in place. Returns (mean training loss of the last epoch, samples processed)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    seen, last_loss = 0, float("nan")
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=generator)
        losses = []
        for start in range(0, len(Xt), batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            losses.append(loss.item())
            seen += len(idx)
        last_loss = float(np.mean(losses))
    return last_loss, seen


def run_federated(
    model_fn: Callable[[], nn.Module],
    clients: list[Client],
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_cfg: dict,
    seed: int = 42,
    on_round: RoundCallback | None = None,
    count_comm: bool = True,
) -> dict:
    """Run FedAvg and return a history with one record per round plus final/per-client metrics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)

    global_model = model_fn()                                     # 1 INITIALISE
    local_models = [model_fn() for _ in clients]
    n_params = count_params(global_model)
    k = max(1, int(round(train_cfg.get("client_fraction", 1.0) * len(clients))))

    history, cum_bytes, samples_seen = [], 0, 0
    t0 = time.perf_counter()
    for r in range(1, train_cfg["rounds"] + 1):                   # 7 REPEAT
        selected = sorted(rng.choice(len(clients), size=k, replace=False).tolist())
        states, weights, losses = [], [], []
        global_state = global_model.state_dict()
        for c in selected:
            local = local_models[c]
            local.load_state_dict(global_state)                   # 2 DISTRIBUTE
            X, y = clients[c]
            loss, seen = train_local(                             # 3 TRAIN LOCALLY
                local, X, y, train_cfg["local_epochs"], train_cfg["batch_size"], train_cfg["lr"], gen
            )
            states.append(local.state_dict())                     # 4 SEND UPDATES
            weights.append(len(y))
            losses.append(loss)
            samples_seen += seen
        global_model.load_state_dict(fedavg(states, weights))     # 5 AGGREGATE + 6 REDISTRIBUTE

        bytes_this_round = round_comm_bytes(n_params, len(selected)) if count_comm else 0
        cum_bytes += bytes_this_round
        test = evaluate(global_model, X_test, y_test)
        record = {
            "round": r,
            "train_loss": float(np.average(losses, weights=weights)),
            **{f"test_{m}": v for m, v in test.items()},
            "clients": [chr(ord("A") + c) for c in selected],
            "comm_bytes": bytes_this_round,
            "cum_comm_bytes": cum_bytes,
            "elapsed_s": time.perf_counter() - t0,
        }
        history.append(record)
        if on_round:
            on_round(record)

    final = evaluate(global_model, X_test, y_test)                # 8 EVALUATE
    per_client = {
        chr(ord("A") + c): evaluate(local_models[c], X_test, y_test)
        for c in range(len(clients))
    }
    return {
        "history": history,
        "final": final,
        "per_client": per_client,
        "n_params": n_params,
        "samples_seen": samples_seen,
        "train_time_s": time.perf_counter() - t0,
        "model": global_model,
    }
