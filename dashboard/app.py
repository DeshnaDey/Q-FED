"""Q-FED dashboard.

    streamlit run dashboard/app.py

Live demo   train a federated VQC in front of the audience, round by round
Results     the six experiments, read from results/ (run experiments/run.py first)
System      architecture, training round and scope of the claims
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Q-FED", page_icon="⚛", layout="wide")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from qfed.config import deep_merge, load_base  # noqa: E402
from qfed.results import history_frame, load_runs, summary_table  # noqa: E402


@st.cache_resource(show_spinner=False)
def quantum_stack() -> SimpleNamespace:
    """Import torch and PennyLane on first use: they are slow to load on a cold server,
    and only the live demo needs them."""
    t0 = time.perf_counter()
    import pennylane as qml

    from qfed.models import build_qnode, circuit_info
    from qfed.runners import run
    print(f"[qfed] loaded torch + PennyLane in {time.perf_counter() - t0:.1f}s", flush=True)
    return SimpleNamespace(qml=qml, build_qnode=build_qnode, circuit_info=circuit_info, run=run)

# ---------------------------------------------------------------- chart styling
# Categorical slots in fixed order; a variant keeps its slot across every chart.
LIGHT = {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
         "grid": "#e1e0d9", "axis": "#c3c2b7", "muted": "#898781"}
DARK = {"series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "grid": "#2c2c2a", "axis": "#383835", "muted": "#898781"}


def palette() -> dict:
    theme = getattr(getattr(st, "context", None), "theme", None)
    return DARK if getattr(theme, "type", "light") == "dark" else LIGHT


def style(fig: go.Figure, title: str, y_title: str, x_title: str, y_range=None, height: int = 340) -> go.Figure:
    p = palette()
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)), height=height, margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified", legend=dict(orientation="h", y=-0.22, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor=p["axis"], tickfont=dict(color=p["muted"]))
    fig.update_yaxes(title=y_title, gridcolor=p["grid"], gridwidth=1, zeroline=False,
                     tickfont=dict(color=p["muted"]), range=y_range)
    return fig


def line(fig: go.Figure, x, y, name: str, slot: int, dash: str | None = None) -> None:
    color = palette()["series"][slot % 8]
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers",
                             line=dict(color=color, width=2, dash=dash), marker=dict(size=8)))


def fmt_bytes(n: float) -> str:
    if n < 1024:
        return f"{int(n)} bytes"
    return f"{n / 1024:.1f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.2f} MB"


# ---------------------------------------------------------------- pages
def page_demo() -> None:
    st.title("Q-FED · live federated training")
    st.caption("Three simulated hospitals train one 4-qubit quantum classifier together. "
               "Only model parameters move between clients and server; raw records stay local.")

    with st.sidebar:
        st.header("Demo settings")
        model_type = st.selectbox("Local model", ["VQC (4 qubits)", "Logistic regression (classical FL)"])
        partition = st.selectbox("Client data split", ["IID", "Non-IID · Dirichlet label skew", "Non-IID · feature skew"])
        alpha = st.slider("Dirichlet α (lower = more skewed)", 0.1, 2.0, 0.5, 0.1,
                          disabled=not partition.startswith("Non-IID · Dirichlet"))
        noise = st.selectbox("Quantum noise", ["Ideal", "Shot noise", "Gate errors (depolarizing)", "Readout errors"],
                             disabled=model_type.startswith("Logistic"))
        shots = st.select_slider("Shots", [50, 100, 500, 1000, 5000], 1000, disabled=noise != "Shot noise")
        p_err = st.select_slider("Error probability p", [0.001, 0.005, 0.01, 0.02, 0.05, 0.1], 0.01,
                                 disabled=noise not in ("Gate errors (depolarizing)", "Readout errors"))
        rounds = st.slider("Communication rounds", 2, 30, 10)
        local_epochs = st.slider("Local epochs per round", 1, 3, 1)
        seed = st.number_input("Seed", 0, 999, 0)

    overrides = {
        "model": {"type": "logreg" if model_type.startswith("Logistic") else "vqc"},
        "data": {"partition": {"IID": "iid", "Non-IID · Dirichlet label skew": "dirichlet",
                               "Non-IID · feature skew": "feature_skew"}[partition], "alpha": alpha},
        "noise": {"mode": {"Ideal": "ideal", "Shot noise": "shots", "Gate errors (depolarizing)": "depolarizing",
                           "Readout errors": "readout"}[noise], "shots": shots, "p": p_err},
        "train": {"rounds": rounds, "local_epochs": local_epochs},
    }
    if overrides["model"]["type"] == "logreg":
        overrides["noise"]["mode"] = "ideal"
    cfg = deep_merge(load_base(), overrides)

    from qfed.data import class_counts, load_data, make_clients  # scikit-learn only, quick to import

    ds = load_data(cfg["data"]["test_size"], cfg["data"]["n_components"], seed=cfg["seed"])
    clients = make_clients(ds, cfg["data"], seed=int(seed))
    counts = pd.DataFrame(class_counts(clients))

    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure()
        s = palette()["series"]
        fig.add_bar(x=counts["client"].map(lambda c: f"Client {c}"), y=counts["benign"], name="Benign",
                    marker_color=s[0], marker_line_width=0)
        fig.add_bar(x=counts["client"].map(lambda c: f"Client {c}"), y=counts["malignant"], name="Malignant",
                    marker_color=s[1], marker_line_width=0)
        fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.06)
        st.plotly_chart(style(fig, "Records held by each client (never shared)", "records", "", height=300),
                        use_container_width=True)
    with right:
        st.markdown("**Training set** · 455 records · 4 PCA features scaled to [0, π]  \n"
                    "**Test set** · 114 held-out records, shared by every run")
        with st.spinner("Loading the quantum simulator (the first load on a cold server can take a minute)…"):
            q = quantum_stack()
        if cfg["model"]["type"] == "vqc":
            info = q.circuit_info(cfg)
            # Draw the noiseless circuit; the active noise is described in the caption instead.
            qnode = q.build_qnode(4, cfg["model"]["n_layers"], {"mode": "ideal"})
            drawing = q.qml.draw(qnode, decimals=None, show_matrices=False, level="device", max_length=100)(
                np.zeros(4), np.zeros((cfg["model"]["n_layers"], 4)))
            st.code(drawing, language=None)
            mode = cfg["noise"]["mode"]
            noise_note = {
                "ideal": "Ideal simulation (exact expectation values).",
                "shots": f"Shot noise: ⟨Z⟩ estimated from {cfg['noise']['shots']} samples per circuit run.",
                "depolarizing": f"Depolarizing error p = {cfg['noise']['p']} after every gate.",
                "readout": f"Readout bit-flip p = {cfg['noise']['p']} on every qubit before measurement.",
            }[mode]
            st.caption(f"Encode (RY) · train (RY) · entangle (CNOT ring) · measure ⟨Z₀⟩ — depth {info['depth']}, "
                       f"{info['quantum_params']} circuit parameters + 2 classical (scale, bias).  \n{noise_note}")
        else:
            st.info("Classical FL baseline: a single linear layer (5 parameters) trained with the same FedAvg loop.")

    if not st.button("▶  Run federated training", type="primary"):
        return

    progress = st.progress(0.0, text="Starting…")
    status = st.empty()
    c1, c2 = st.columns(2)
    acc_slot, loss_slot = c1.empty(), c2.empty()
    rows: list[dict] = []

    def on_round(rec: dict) -> None:
        rows.append(rec)
        df = pd.DataFrame(rows)
        progress.progress(rec["round"] / rounds, text=f"Round {rec['round']} of {rounds}")
        status.markdown(f"Round **{rec['round']}** · clients {', '.join(rec['clients'])} trained locally · "
                        f"{fmt_bytes(rec['comm_bytes'])} of parameters exchanged · "
                        f"test accuracy **{rec['test_accuracy']:.3f}**")
        f1 = go.Figure()
        line(f1, df["round"], df["test_accuracy"], "Accuracy", 0)
        line(f1, df["round"], df["test_f1"], "F1 (malignant)", 1)
        acc_slot.plotly_chart(style(f1, "Global model on the held-out test set", "score", "round", [0, 1.02]),
                              use_container_width=True, key=f"acc{rec['round']}")
        f2 = go.Figure()
        line(f2, df["round"], df["train_loss"], "Train loss (weighted mean of clients)", 0)
        line(f2, df["round"], df["test_loss"], "Test loss", 1)
        loss_slot.plotly_chart(style(f2, "Loss", "BCE loss", "round"), use_container_width=True,
                               key=f"loss{rec['round']}")

    with st.spinner("Training…"):
        result = q.run(cfg, "federated", seed=int(seed), on_round=on_round)
    progress.progress(1.0, text="Done")

    f = result["final"]
    m = st.columns(5)
    m[0].metric("Accuracy", f"{f['accuracy']:.3f}")
    m[1].metric("F1 (malignant)", f"{f['f1']:.3f}")
    m[2].metric("ROC-AUC", f"{f['roc_auc']:.3f}")
    m[3].metric("Parameters exchanged", fmt_bytes(result["history"][-1]["cum_comm_bytes"]))
    m[4].metric("Training time", f"{result['train_time_s']:.1f} s")

    table = pd.DataFrame({"Global (FedAvg)": f, **{f"Client {k} · last local update": v
                                                   for k, v in result["per_client"].items()}}).T
    st.subheader("Global vs client-level performance on the test set")
    st.dataframe(table[["accuracy", "precision", "recall", "f1", "roc_auc"]].style.format("{:.3f}"),
                 use_container_width=True)
    st.caption(f"Client rows score each client's own parameters after its final local training, before averaging.  \n"
               f"Raw records transferred: **0**. Each round every participating client sent "
               f"{result['n_params']} numbers ({fmt_bytes(result['n_params'] * 4)}) to the server.")


def page_results() -> None:
    st.title("Experiment results")
    runs = load_runs()
    if not runs:
        st.warning("No results yet. Run the experiments first:")
        st.code("python experiments/run.py --all --quick      # ~minutes, 1 seed\n"
                "python experiments/run.py --all --seeds 3    # full run", language="bash")
        return

    summary, hist = summary_table(runs), history_frame(runs)
    titles = {r["experiment"]: r["experiment_title"] for r in runs}
    questions = {r["experiment"]: r["question"] for r in runs}
    exp = st.selectbox("Experiment", sorted(titles), format_func=titles.get)
    st.markdown(f"**Question:** {questions[exp]}")

    s = summary[summary["experiment"] == exp]
    metric = st.radio("Metric", ["accuracy", "f1", "roc_auc", "precision", "recall"], horizontal=True,
                      format_func=lambda x: {"roc_auc": "ROC-AUC", "f1": "F1"}.get(x, x.title()))

    fig = go.Figure(go.Bar(
        y=s["label"], x=s[f"{metric}_mean"], orientation="h",
        error_x=dict(type="data", array=s[f"{metric}_std"].fillna(0), color=palette()["muted"], thickness=1.5),
        marker_color=palette()["series"][0], marker_line_width=0,
        text=s[f"{metric}_mean"].map("{:.3f}".format), textposition="inside", insidetextanchor="start",
        textfont=dict(color="#ffffff"),
        hovertemplate="%{y}<br>%{x:.3f}<extra></extra>",
    ))
    fig.update_layout(bargap=0.4, hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    lo = max(0.0, float(np.nanmin(s[f"{metric}_mean"])) - 0.15)
    style(fig, f"Final {metric.replace('_', '-')} on the held-out test set (mean ± std across seeds)",
          "", metric, height=80 + 55 * len(s))
    fig.update_xaxes(range=[lo, 1.05], showgrid=True, gridcolor=palette()["grid"])
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    h = hist[hist["experiment"] == exp]
    if not h.empty:
        mean = h.groupby(["order", "label", "round"], as_index=False).mean(numeric_only=True)
        c1, c2 = st.columns(2)
        f1 = go.Figure()
        for order, g in mean.groupby("order"):
            line(f1, g["round"], g["test_accuracy"], g["label"].iloc[0], int(order))
        c1.plotly_chart(style(f1, "Test accuracy per round", "accuracy", "communication round"),
                        use_container_width=True)
        f2 = go.Figure()
        for order, g in mean.groupby("order"):
            line(f2, g["round"], g["train_loss"], g["label"].iloc[0], int(order))
        c2.plotly_chart(style(f2, "Training loss per round", "BCE loss", "communication round"),
                        use_container_width=True)
        if mean["cum_comm_bytes"].max() > 0:
            f3 = go.Figure()
            for order, g in mean.groupby("order"):
                if g["cum_comm_bytes"].max() > 0:
                    line(f3, g["cum_comm_bytes"] / 1024, g["test_accuracy"], g["label"].iloc[0], int(order))
            st.plotly_chart(style(f3, "Accuracy vs parameters exchanged", "accuracy", "cumulative KB sent"),
                            use_container_width=True)

    st.subheader("Summary table")
    show = s[["label", "seeds"] + [f"{m}_{k}" for m in ["accuracy", "f1", "roc_auc"] for k in ["mean", "std"]]]
    st.dataframe(show.rename(columns={"label": "setup"}).style.format(precision=3), use_container_width=True,
                 hide_index=True)
    st.caption("All setups are scored on the same 114-record held-out test set. "
               "Results are reported as measured, including where classical baselines win.")


def page_system() -> None:
    st.title("How Q-FED works")
    st.markdown("""
**Research question.** Can a federated VQC reach useful accuracy on IID and non-IID data under simulated
noise, and how does it compare with classical and centralized baselines?

| Layer | What it does |
|---|---|
| Data | Breast Cancer Wisconsin (569 × 30) → stratified 80/20 split → StandardScaler → PCA(4) → [0, π]; training set split across 3 clients |
| Clients A, B, C | Each trains the same 4-qubit VQC on its own records only |
| Server | FedAvg: θ_global = Σ_k (n_k / n) · θ_k, sent back each round |
| Noise | Shot noise (finite sampling), depolarizing gate errors, readout bit-flips |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, loss, communication, circuit cost |

**One round:** distribute θ → train locally → send θ_k (never records) → weighted average → redistribute.

**Scope.** This is a feasibility study in simulation on a normal computer. It does not claim quantum advantage,
and the privacy claim is limited to training without transferring raw records: model updates can still leak
information, so differential privacy and secure aggregation are future work. The dataset is used only as a
technical demonstration, not for medical decisions.
""")


PAGES = {"Live demo": page_demo, "Experiment results": page_results, "How it works": page_system}
with st.sidebar:
    st.markdown("### ⚛ Q-FED")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.divider()
PAGES[page]()
