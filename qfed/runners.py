"""One entry point per training setup compared in the experiments.

federated     FedAvg across all clients (quantum or classical, set by model.type)
local_only    each client trains alone for the same number of epochs; no collaboration
centralized   one model on the pooled training set (a single-client federation, no comm cost)
sklearn_lr    scikit-learn logistic regression on the pooled training set
sklearn_svm   scikit-learn RBF SVM on the pooled training set
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qfed.data import Dataset, class_counts, load_data, make_clients
from qfed.fedavg import RoundCallback, run_federated
from qfed.metrics import classification_metrics
from qfed.models import circuit_info, make_model

RUNNERS = ("federated", "local_only", "centralized", "sklearn_lr", "sklearn_svm")


def _strip(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "model"}


def run(cfg: dict, runner: str, seed: int, on_round: RoundCallback | None = None) -> dict:
    """Run one setup with one seed and return a JSON-serialisable result."""
    ds: Dataset = load_data(cfg["data"]["test_size"], cfg["data"]["n_components"], seed=cfg["seed"])
    clients = make_clients(ds, cfg["data"], seed=seed)
    model_fn = lambda: make_model(cfg)  # noqa: E731
    train = cfg["train"]

    if runner == "federated":
        result = _strip(run_federated(model_fn, clients, ds.X_test, ds.y_test, train, seed, on_round))

    elif runner == "centralized":
        result = _strip(run_federated(
            model_fn, [(ds.X_train, ds.y_train)], ds.X_test, ds.y_test,
            {**train, "client_fraction": 1.0}, seed, on_round, count_comm=False,
        ))

    elif runner == "local_only":
        # Each client trains alone; history and final metrics are the mean across clients.
        runs = [
            run_federated(model_fn, [c], ds.X_test, ds.y_test, {**train, "client_fraction": 1.0},
                          seed + i, None, count_comm=False)
            for i, c in enumerate(clients)
        ]
        metric_keys = [k for k in runs[0]["history"][0] if k.startswith("test_") or k == "train_loss"]
        history = []
        for r in range(train["rounds"]):
            rec = {"round": r + 1, "comm_bytes": 0, "cum_comm_bytes": 0, "clients": []}
            rec.update({k: float(np.nanmean([run_["history"][r][k] for run_ in runs])) for k in metric_keys})
            history.append(rec)
            if on_round:
                on_round(rec)
        result = {
            "history": history,
            "final": {k: float(np.nanmean([r["final"][k] for r in runs])) for k in runs[0]["final"]},
            "per_client": {chr(ord("A") + i): r["final"] for i, r in enumerate(runs)},
            "n_params": runs[0]["n_params"],
            "samples_seen": sum(r["samples_seen"] for r in runs),
            "train_time_s": sum(r["train_time_s"] for r in runs),
        }

    elif runner in ("sklearn_lr", "sklearn_svm"):
        clf = (LogisticRegression(max_iter=1000, random_state=seed) if runner == "sklearn_lr"
               else CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), ensemble=False))
        t0 = time.perf_counter()
        clf.fit(ds.X_train, ds.y_train)
        final = classification_metrics(ds.y_test, clf.predict_proba(ds.X_test)[:, 1])
        result = {"history": [], "final": final, "per_client": {}, "n_params": 0,
                  "samples_seen": len(ds.y_train), "train_time_s": time.perf_counter() - t0}

    else:
        raise ValueError(f"Unknown runner: {runner}")

    result["client_counts"] = class_counts(clients)
    is_sklearn = runner in ("sklearn_lr", "sklearn_svm")
    result["circuit"] = circuit_info({"model": {"type": "classical"}} if is_sklearn else cfg)
    # Hardware-equivalent circuit runs: 1 forward + 2 per quantum parameter (parameter shift) per sample.
    q = result["circuit"]["quantum_params"]
    result["circuit"]["est_circuit_evals"] = int(result["samples_seen"] * (1 + 2 * q)) if q else 0
    return result
