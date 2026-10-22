# Q-FED: Noise-Resilient Hybrid Quantum Federated Learning

A simulation prototype in which three clients train one 4-qubit Variational Quantum Classifier (VQC)
together with Federated Averaging, without sharing raw records. It measures how the federated VQC
behaves under non-IID client data and simulated quantum noise, and compares it with centralized,
local-only and classical baselines.

Dataset: Breast Cancer Wisconsin (Diagnostic), 569 records × 30 features, from scikit-learn. It is used
only as a technical demonstration, not for medical decisions.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # app + pytest
```

## Run the demo

```bash
streamlit run dashboard/app.py
```

The dashboard opens at http://localhost:8501 with three pages:

- **Live demo**: pick the data split, noise model and number of rounds, press *Run federated training*,
  and watch the global model's test accuracy and loss update each round. Shows each client's class mix,
  the circuit, final metrics, global vs client-level scores and the bytes exchanged.
- **Experiment results**: the six experiments, from the files in `results/`.
- **How it works**: architecture, the training round and the scope of the claims.

## Run the experiments

```bash
python experiments/run.py --all --quick         # smoke test: 5 rounds, 1 seed
python experiments/run.py --all --seeds 3       # full run, mean ± std over 3 seeds
python experiments/run.py configs/exp5_noise_sensitivity.yaml
```

| # | Config | Compares |
|---|---|---|
| 1 | `exp1_local_vs_federated.yaml` | local-only vs FedAvg, IID and non-IID |
| 2 | `exp2_iid_vs_noniid.yaml` | IID vs Dirichlet α ∈ {1.0, 0.5, 0.1} vs feature skew |
| 3 | `exp3_centralized_vs_federated.yaml` | pooled-data VQC vs federated VQC |
| 4 | `exp4_quantum_vs_classical.yaml` | VQC vs logistic regression (federated and centralized) vs SVM |
| 5 | `exp5_noise_sensitivity.yaml` | ideal vs shot noise vs gate errors vs readout errors |
| 6 | `exp6_communication_rounds.yaml` | convergence over 40 rounds, client participation, local epochs |

Each run writes `results/<experiment>/<variant>__seed<k>.json`. Shared defaults are in `configs/base.yaml`.

## Tests

```bash
pytest
```

## Layout

```
qfed/data.py       load, split, StandardScaler → PCA(4) → [0, π], IID / Dirichlet / feature-skew partitions
qfed/noise.py      simulator device per noise mode (ideal, shots, depolarizing, readout)
qfed/models.py     4-qubit VQC (RY encoding, RY layers, CNOT ring, ⟨Z₀⟩) and logistic-regression baseline
qfed/fedavg.py     the FedAvg loop, step by step
qfed/runners.py    federated, local-only, centralized, scikit-learn baselines
qfed/metrics.py    accuracy, precision, recall, F1, ROC-AUC, communication cost
qfed/results.py    JSON storage and summary tables
experiments/run.py experiment runner
dashboard/app.py   Streamlit + Plotly dashboard
```

## Scope

This is a feasibility study in simulation. It does not claim quantum advantage. The privacy claim is limited
to training without transferring raw records; model updates can still leak information, so differential
privacy and secure aggregation are future work.
