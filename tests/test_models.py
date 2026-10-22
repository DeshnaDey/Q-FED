import numpy as np
import torch

from qfed.data import load_data
from qfed.fedavg import run_federated
from qfed.models import VQCClassifier
from qfed.data import make_clients


def test_vqc_forward_shape():
    model = VQCClassifier(4, 2)
    out = model(torch.rand(5, 4))
    assert out.shape == (5,)


def test_vqc_noisy_modes_run():
    x = torch.rand(3, 4)
    for noise in ({"mode": "depolarizing", "p": 0.05}, {"mode": "readout", "p": 0.05}):
        assert VQCClassifier(4, 1, noise)(x).shape == (3,)


def test_federated_vqc_learns():
    ds = load_data(seed=0)
    clients = make_clients(ds, {"n_clients": 3, "partition": "iid"}, seed=0)
    cfg = {"rounds": 3, "local_epochs": 1, "batch_size": 16, "lr": 0.05}
    res = run_federated(lambda: VQCClassifier(4, 2), clients, ds.X_test, ds.y_test, cfg, seed=0)
    assert len(res["history"]) == 3
    assert res["final"]["accuracy"] > 0.75
