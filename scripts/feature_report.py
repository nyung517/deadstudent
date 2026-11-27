"""Which features carry the dropout signal?  Coefficients first, importances second.

Two views, both interpreted at the *source feature* level (the project forbids PCA):

1. LogisticRegression(max_iter=2000, class_weight="balanced") on the design matrix
   build_design produces.  One-hot columns are mapped back to the feature they came from
   and the top 10 by |coef| are printed with the odds ratio exp(coef).
   Numeric features are standardised by build_design, so their odds ratio is per
   one standard deviation; one-hot odds ratios are per dummy vs the dropped level.
2. RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
   class_weight="balanced", random_state=42).  Its per-column importances are
   summed per source feature, which is the fair comparison between a numeric
   feature (1 column) and a categorical one (many columns).

Run:
    uv run python scripts/feature_report.py            # full train part
    uv run python scripts/feature_report.py --smoke    # tiny subsample, smoke test
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from deadstudent.data import RAW_PATH, SEED, load_raw, split_frame
from deadstudent.features import Design, build_design, feature_columns

CSV_PATH = Path("artifacts/feature_importance.csv")
FIG_PATH = Path("artifacts/figures/feature_importance.png")
TOP_COEFS = 10
TOP_FEATURES = 12
SMOKE_TRAIN_ROWS = 400


def source_feature(encoded_name: str, sources: list[str]) -> str:
    """Map 'nominal__course_9500' -> 'course' and 'numeric__gdp' -> 'gdp'.

    The longest matching source name wins, so 'numeric__previous_qualification_grade'
    keeps its full name instead of collapsing into 'previous_qualification'.
    """
    raw = encoded_name.split("__", 1)[1] if "__" in encoded_name else encoded_name
    hits = [name for name in sources if raw == name or raw.startswith(f"{name}_")]
    return max(hits, key=len) if hits else raw


def logistic_table(design: Design, sources: list[str]) -> pd.DataFrame:
    """Per-column logistic coefficients with the source feature attached."""
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(design.X_train, design.y_train)
    coefs = model.coef_[0]
    table = pd.DataFrame(
        {
            "encoded_column": design.feature_names,
            "source_feature": [source_feature(name, sources) for name in design.feature_names],
            "coef": coefs,
        }
    )
    table["odds_ratio"] = np.exp(table["coef"])
    table["abs_coef"] = table["coef"].abs()
    return table.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def forest_importance(design: Design, sources: list[str]) -> pd.DataFrame:
    """Random-forest importances summed per source feature."""
    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(design.X_train, design.y_train)
    per_column = pd.DataFrame(
        {
            "source_feature": [source_feature(name, sources) for name in design.feature_names],
            "importance": model.feature_importances_,
        }
    )
    table = (
        per_column.groupby("source_feature", as_index=False)
        .agg(importance=("importance", "sum"), n_columns=("importance", "size"))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["rank", "source_feature", "importance", "n_columns"]]


def save_importance_figure(table: pd.DataFrame, path: Path = FIG_PATH) -> Path:
    """Horizontal bars for the top features, sorted by importance."""
    top = table.head(TOP_FEATURES).iloc[::-1]
    figure, axes = plt.subplots(figsize=(7.5, 5.0))
    axes.barh(top["source_feature"], top["importance"], color="#4f81bd")
    for value, label in zip(top["importance"], top["source_feature"]):
        axes.text(value, label, f" {value:.4f}", va="center", fontsize=8)
    axes.set_xlabel("summed random-forest importance")
    axes.set_title(f"Top {len(top)} source features")
    axes.set_xlim(0, top["importance"].max() * 1.18)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="coefficient and importance report")
    parser.add_argument("--smoke", action="store_true", help="tiny train subsample, writes *_smoke artifacts")
    args = parser.parse_args()

    split = split_frame(load_raw(RAW_PATH))
    note = "full train part"
    if args.smoke:
        split = type(split)(
            train=split.train.head(SMOKE_TRAIN_ROWS).reset_index(drop=True),
            valid=split.valid,
            test=split.test,
        )
        note = f"SMOKE subsample (train={SMOKE_TRAIN_ROWS})"

    design = build_design(split)
    sources = feature_columns(split.train)

    coefs = logistic_table(design, sources)
    importance = forest_importance(design, sources)

    csv_path = CSV_PATH.with_name(f"{CSV_PATH.stem}_smoke.csv") if args.smoke else CSV_PATH
    fig_path = FIG_PATH.with_name(f"{FIG_PATH.stem}_smoke{FIG_PATH.suffix}") if args.smoke else FIG_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    importance.round(6).to_csv(csv_path, index=False)
    save_importance_figure(importance, fig_path)

    print(f"feature report: {note}")
    print(f"design matrix: {design.X_train.shape[0]} rows x {design.X_train.shape[1]} one-hot columns "
          f"from {len(set(coefs['source_feature']))} source features")
    print("")
    print(f"LogisticRegression(max_iter=2000, class_weight='balanced') - top {TOP_COEFS} by |coef|")
    print(f"{'rank':>4}  {'source_feature':<34} {'encoded_column':<44} {'coef':>9} {'odds_ratio':>11}")
    for rank, row in coefs.head(TOP_COEFS).iterrows():
        print(f"{rank + 1:>4}  {row['source_feature']:<34.34} {row['encoded_column']:<44.44} "
              f"{row['coef']:>9.4f} {row['odds_ratio']:>11.4f}")
    print("odds_ratio = exp(coef); >1 raises dropout odds, <1 lowers them")
    print("")
    print(f"RandomForest(n_estimators=400, min_samples_leaf=5, class_weight='balanced', random_state={SEED})"
          f" - importances summed per source feature, top {TOP_FEATURES} of {len(importance)}")
    printable = importance.head(TOP_FEATURES).copy()
    printable["importance"] = printable["importance"].round(4)
    print(printable.to_string(index=False))
    print(f"importance = sum of feature_importances_ over the {int(importance['n_columns'].sum())} one-hot columns; "
          f"totals {importance['importance'].sum():.4f}")
    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
