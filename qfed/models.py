"""Local models: the hybrid 4-qubit VQC and a classical logistic-regression baseline.

Both are plain torch modules that output logits, so the same FedAvg loop trains either.
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

from qfed.noise import make_device, with_shots


def build_qnode(n_qubits: int, n_layers: int, noise_cfg: dict):
    dev, diff_method, shots = make_device(n_qubits, noise_cfg)
    mode = noise_cfg.get("mode", "ideal")
    p = float(noise_cfg.get("p", 0.0))

    def gate_noise(wires):
        if mode == "depolarizing":
            for w in wires:
                qml.DepolarizingChannel(p, wires=w)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # ENCODE: one scaled feature per qubit as an RY angle
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        gate_noise(range(n_qubits))
        for layer in range(n_layers):
            # TRAIN: trainable RY on every qubit
            for w in range(n_qubits):
                qml.RY(weights[layer, w], wires=w)
            gate_noise(range(n_qubits))
            # ENTANGLE: CNOT ring between neighbouring qubits
            for w in range(n_qubits):
                target = (w + 1) % n_qubits
                qml.CNOT(wires=[w, target])
                gate_noise([w, target])
        if mode == "readout":
            for w in range(n_qubits):
                qml.BitFlip(p, wires=w)
        # MEASURE
        return qml.expval(qml.PauliZ(0))

    return with_shots(circuit, shots)


class VQCClassifier(nn.Module):
    """Hybrid classifier: logit = a * <Z0> + b, with the circuit weights and (a, b) all trainable."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, noise_cfg: dict | None = None):
        super().__init__()
        noise_cfg = noise_cfg or {"mode": "ideal"}
        qnode = build_qnode(n_qubits, n_layers, noise_cfg)
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits)})
        self.scale = nn.Parameter(torch.tensor(-2.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.qlayer(x).float().reshape(-1)
        return self.scale * z + self.bias


class LogRegClassifier(nn.Module):
    """Classical baseline for federated training: a single linear layer."""

    def __init__(self, n_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).reshape(-1)


def make_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m["type"] == "vqc":
        return VQCClassifier(m["n_qubits"], m["n_layers"], cfg["noise"])
    if m["type"] == "logreg":
        return LogRegClassifier(cfg["data"]["n_components"])
    raise ValueError(f"Unknown model type: {m['type']}")


def circuit_info(cfg: dict) -> dict:
    """Static quantum cost of the VQC (zeros for classical models)."""
    m = cfg["model"]
    if m["type"] != "vqc":
        return {"qubits": 0, "depth": 0, "quantum_params": 0, "shots": 0}
    n, L = m["n_qubits"], m["n_layers"]
    # embedding (1) + per layer: RY (1) + CNOT ring (n sequential gates)
    depth = 1 + L * (1 + n)
    shots = cfg["noise"]["shots"] if cfg["noise"]["mode"] == "shots" else 0
    return {"qubits": n, "depth": depth, "quantum_params": n * L, "shots": shots}
