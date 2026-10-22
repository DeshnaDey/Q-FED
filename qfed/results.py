"""Results storage: one JSON file per (experiment, variant, seed) under results/."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from qfed.config import ROOT

RESULTS_DIR = ROOT / "results"


def _clean(obj):
    """Replace NaN with None so the files are valid JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def save_run(experiment: str, variant: str, seed: int, payload: dict, root: Path = RESULTS_DIR) -> Path:
    path = root / experiment / f"{variant}__seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2))
    return path


def load_runs(root: Path = RESULTS_DIR) -> list[dict]:
    if not root.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(root.glob("*/*.json"))]


def summary_table(runs: list[dict]) -> pd.DataFrame:
    """Final metrics, mean and std across seeds, one row per (experiment, variant)."""
    rows = [{"experiment": r["experiment"], "variant": r["variant"], "label": r.get("label", r["variant"]),
             "order": r.get("order", 0), "seed": r["seed"], **r["final"]} for r in runs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    agg = df.groupby(["experiment", "order", "variant", "label"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["seeds"] = df.groupby(["experiment", "order", "variant", "label"]).size()
    return agg.reset_index().sort_values(["experiment", "order"])


def history_frame(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        for h in r.get("history", []):
            rows.append({"experiment": r["experiment"], "variant": r["variant"],
                         "label": r.get("label", r["variant"]), "order": r.get("order", 0),
                         "seed": r["seed"], **{k: v for k, v in h.items() if k != "clients"}})
    return pd.DataFrame(rows)
