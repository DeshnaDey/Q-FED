"""Noise layer: choose the simulator device, gradient method and shot count for each noise mode."""

from __future__ import annotations

import pennylane as qml

NOISE_MODES = ("ideal", "shots", "depolarizing", "readout")


def make_device(n_qubits: int, noise_cfg: dict):
    """Return (device, diff_method, shots).

    ideal         analytic state-vector simulation, backprop gradients
    shots         finite-shot sampling; needs parameter-shift (as on real hardware)
    depolarizing  density-matrix simulation with a depolarizing channel after every gate
    readout       density-matrix simulation with bit-flip errors before measurement
    """
    mode = noise_cfg.get("mode", "ideal")
    if mode == "ideal":
        return qml.device("default.qubit", wires=n_qubits), "backprop", None
    if mode == "shots":
        return qml.device("default.qubit", wires=n_qubits), "parameter-shift", int(noise_cfg["shots"])
    if mode in ("depolarizing", "readout"):
        return qml.device("default.mixed", wires=n_qubits), "backprop", None
    raise ValueError(f"Unknown noise mode: {mode}")


def with_shots(qnode, shots: int | None):
    """Attach a finite shot count to a QNode (qml.set_shots on PennyLane >= 0.42)."""
    if not shots:
        return qnode
    if hasattr(qml, "set_shots"):
        return qml.set_shots(qnode, shots=shots)
    return qnode.update_shots(shots)
