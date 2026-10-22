"""Run one or all experiments and write results to results/<experiment>/.

    python experiments/run.py configs/exp2_iid_vs_noniid.yaml
    python experiments/run.py --all --seeds 3
    python experiments/run.py --all --quick          # 5 rounds, 1 seed: a fast smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qfed.config import ROOT, deep_merge, load_base, load_yaml  # noqa: E402
from qfed.results import save_run  # noqa: E402
from qfed.runners import run  # noqa: E402


def run_experiment(path: Path, seeds: int, quick: bool) -> None:
    exp = load_yaml(path)
    base = deep_merge(load_base(), exp.get("overrides", {}))
    if quick:
        base = deep_merge(base, {"train": {"rounds": min(5, base["train"]["rounds"])}})
    print(f"\n=== {exp['title']}  —  {exp['question']}")
    for order, variant in enumerate(exp["variants"]):
        cfg = deep_merge(base, variant.get("overrides", {}))
        n_seeds = 1 if variant["runner"].startswith("sklearn") and not quick else seeds
        for seed in range(n_seeds):
            t0 = time.perf_counter()
            result = run(cfg, variant["runner"], seed=seed)
            payload = {
                "experiment": exp["name"], "experiment_title": exp["title"], "question": exp["question"],
                "variant": variant["name"], "label": variant.get("label", variant["name"]),
                "runner": variant["runner"], "order": order, "seed": seed, "config": cfg, **result,
            }
            save_run(exp["name"], variant["name"], seed, payload)
            f = result["final"]
            print(f"  {variant['label']:<34} seed {seed}  acc {f['accuracy']:.3f}  f1 {f['f1']:.3f}"
                  f"  auc {f['roc_auc']:.3f}  ({time.perf_counter() - t0:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", type=Path, help="experiment yaml files")
    ap.add_argument("--all", action="store_true", help="run every configs/exp*.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="5 rounds and 1 seed")
    args = ap.parse_args()

    paths = sorted((ROOT / "configs").glob("exp*.yaml")) if args.all else args.configs
    if not paths:
        ap.error("pass experiment config files or --all")
    seeds = 1 if args.quick else args.seeds
    for p in paths:
        run_experiment(p, seeds, args.quick)


if __name__ == "__main__":
    main()
