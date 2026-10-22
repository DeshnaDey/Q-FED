"""Data layer: load, split, reduce, scale and partition the Breast Cancer Wisconsin dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_data(test_size: float = 0.2, n_components: int = 4, seed: int = 42) -> Dataset:
    """Stratified split, then StandardScaler -> PCA -> MinMax to [0, pi], fitted on train only.

    Labels are flipped so malignant = 1 is the positive class (sklearn uses 0 = malignant).
    """
    X, y = load_breast_cancer(return_X_y=True)
    y = 1 - y
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    prep = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        MinMaxScaler(feature_range=(0, np.pi)),
    )
    X_tr = prep.fit_transform(X_tr)
    X_te = prep.transform(X_te)
    return Dataset(
        X_tr.astype(np.float32), y_tr.astype(np.float32),
        X_te.astype(np.float32), y_te.astype(np.float32),
    )


def partition(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    scheme: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
    min_client_size: int = 10,
) -> list[np.ndarray]:
    """Split training indices across clients. Returns one index array per client."""
    rng = np.random.default_rng(seed)

    if scheme == "iid":
        skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
        return [np.sort(idx) for _, idx in skf.split(X, y)]

    if scheme == "feature_skew":
        # Each client sees a different region of the first principal component.
        order = np.argsort(X[:, 0], kind="stable")
        return [np.sort(chunk) for chunk in np.array_split(order, n_clients)]

    if scheme == "dirichlet":
        # Label skew: per class, draw client shares from Dir(alpha).
        for _ in range(1000):
            shards: list[list[int]] = [[] for _ in range(n_clients)]
            for cls in np.unique(y):
                idx = rng.permutation(np.where(y == cls)[0])
                shares = rng.dirichlet(alpha * np.ones(n_clients))
                cuts = (np.cumsum(shares)[:-1] * len(idx)).astype(int)
                for k, part in enumerate(np.split(idx, cuts)):
                    shards[k].extend(part.tolist())
            if min(len(s) for s in shards) >= min_client_size:
                return [np.sort(np.array(s)) for s in shards]
        raise RuntimeError("Could not draw a Dirichlet split meeting min_client_size")

    raise ValueError(f"Unknown partition scheme: {scheme}")


def make_clients(ds: Dataset, data_cfg: dict, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    shards = partition(
        ds.X_train, ds.y_train,
        n_clients=data_cfg["n_clients"],
        scheme=data_cfg["partition"],
        alpha=data_cfg.get("alpha", 0.5),
        seed=seed,
        min_client_size=data_cfg.get("min_client_size", 10),
    )
    return [(ds.X_train[idx], ds.y_train[idx]) for idx in shards]


def class_counts(clients: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Benign / malignant counts per client, for logging and the dashboard."""
    return [
        {"client": chr(ord("A") + k), "benign": int((y == 0).sum()), "malignant": int((y == 1).sum())}
        for k, (_, y) in enumerate(clients)
    ]
