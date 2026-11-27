"""Depth sweep for a single decision tree.

Question the sweep answers: at what ``max_depth`` does the tree stop learning the
signal and start memorising the training rows?  For every depth in 1..20 we refit
the tree on the *same* train part and record F1/AUC on train and on valid, so the
two curves can be read side by side.

``class_weight="balanced"`` and ``random_state`` mirror ``scripts/run_models.py``.
``min_samples_leaf`` defaults to 1 here so that a deep tree can actually memorise
the training rows; with the project's ``min_samples_leaf=20`` the tree stops
growing around depth 7 and the curves go flat.  Pass ``--min-samples-leaf 20`` to
reproduce the project tree exactly.

Run:
    uv run python scripts/overfit_sweep.py            # full train part
    uv run python scripts/overfit_sweep.py --smoke    # tiny subsample, smoke test
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from deadstudent.data import RAW_PATH, SEED, load_raw, split_frame
from deadstudent.features import Design, build_design

DEPTHS: tuple[int, ...] = tuple(range(1, 21))
CSV_PATH = Path("artifacts/overfit_depth.csv")
FIG_PATH = Path("artifacts/figures/overfit_depth.png")
MIN_SAMPLES_LEAF = 1
THRESHOLD = 0.5

SMOKE_TRAIN_ROWS = 300
SMOKE_VALID_ROWS = 120


def sweep_depth(
    design: Design,
    depths: tuple[int, ...] = DEPTHS,
    min_samples_leaf: int = MIN_SAMPLES_LEAF,
) -> pd.DataFrame:
    """Fit one tree per depth and return train/valid F1 and AUC per depth."""
    y_train = np.asarray(design.y_train)
    y_valid = np.asarray(design.y_valid)
    rows: list[dict[str, float]] = []
    for depth in depths:
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=SEED,
        )
        model.fit(design.X_train, design.y_train)
        train_scores = model.predict_proba(design.X_train)[:, 1]
        valid_scores = model.predict_proba(design.X_valid)[:, 1]
        rows.append(
            {
                "max_depth": depth,
                "train_f1": f1_score(y_train, (train_scores >= THRESHOLD).astype(int), zero_division=0),
                "valid_f1": f1_score(y_valid, (valid_scores >= THRESHOLD).astype(int), zero_division=0),
                "train_auc": roc_auc_score(y_train, train_scores),
                "valid_auc": roc_auc_score(y_valid, valid_scores),
            }
        )
    return pd.DataFrame(rows)


def save_curve(table: pd.DataFrame, path: Path = FIG_PATH) -> Path:
    """Two panels (F1, AUC); each panel has a train line and a valid line."""
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    panels = (("f1", "F1 at threshold 0.5"), ("auc", "ROC AUC"))
    for axis, (metric, title) in zip(axes, panels):
        axis.plot(table["max_depth"], table[f"train_{metric}"], marker="o", label="train", color="#4f81bd")
        axis.plot(table["max_depth"], table[f"valid_{metric}"], marker="o", label="valid", color="#c0504d")
        axis.set_title(title)
        axis.set_xlabel("max_depth")
        axis.set_ylabel(metric.upper())
        axis.set_xticks(list(table["max_depth"]))
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("Decision tree depth sweep: train vs valid")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="max_depth sweep for one decision tree")
    parser.add_argument("--smoke", action="store_true", help="tiny train/valid subsample, writes *_smoke artifacts")
    parser.add_argument("--max-depth", type=int, default=20, help="highest depth to try (default 20)")
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=MIN_SAMPLES_LEAF,
        help="leaf size; 20 matches the project tree, 1 lets deep trees overfit (default 1)",
    )
    args = parser.parse_args()

    depths = tuple(range(1, args.max_depth + 1))
    split = split_frame(load_raw(RAW_PATH))
    note = "full train part"
    if args.smoke:
        split = type(split)(
            train=split.train.head(SMOKE_TRAIN_ROWS).reset_index(drop=True),
            valid=split.valid.head(SMOKE_VALID_ROWS).reset_index(drop=True),
            test=split.test,
        )
        note = f"SMOKE subsample (train={SMOKE_TRAIN_ROWS}, valid={SMOKE_VALID_ROWS})"

    design = build_design(split)
    table = sweep_depth(design, depths, min_samples_leaf=args.min_samples_leaf)

    csv_path = CSV_PATH.with_name(f"{CSV_PATH.stem}_smoke.csv") if args.smoke else CSV_PATH
    fig_path = FIG_PATH.with_name(f"{FIG_PATH.stem}_smoke{FIG_PATH.suffix}") if args.smoke else FIG_PATH

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(csv_path, index=False)
    save_curve(table, fig_path)

    print(f"depth sweep: {len(table)} fits, {note}")
    print(
        f"train n={len(design.y_train)} positives={int(np.asarray(design.y_train).sum())} | "
        f"valid n={len(design.y_valid)} positives={int(np.asarray(design.y_valid).sum())}"
    )
    print(
        "tree: DecisionTreeClassifier(class_weight='balanced', "
        f"min_samples_leaf={args.min_samples_leaf}, random_state={SEED})"
    )
    printable = table.copy()
    printable["f1_gap"] = printable["train_f1"] - printable["valid_f1"]
    print(printable.round(4).to_string(index=False))
    best = table.loc[table["valid_f1"].idxmax()]
    print(f"best valid F1 on this run: {best['valid_f1']:.4f} at max_depth={int(best['max_depth'])}")
    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
