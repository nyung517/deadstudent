from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from deadstudent.data import load_raw

FIG_DIR = Path("artifacts/figures")
OUTCOME_ORDER = ["Dropout", "Enrolled", "Graduate"]


def save_outcome_balance(frame: pd.DataFrame) -> Path:
    counts = frame["outcome"].value_counts().reindex(OUTCOME_ORDER)
    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    bars = axes.bar(counts.index, counts.to_numpy(), color=["#c0504d", "#9bbb59", "#4f81bd"])
    for bar, value in zip(bars, counts.to_numpy()):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 40, f"{value}", ha="center", fontsize=10)
    axes.set_ylabel("students")
    axes.set_title("Outcome label balance (n=%d)" % len(frame))
    axes.set_ylim(0, counts.max() * 1.12)
    figure.tight_layout()
    path = FIG_DIR / "outcome_balance.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_approved_spread(frame: pd.DataFrame) -> Path:
    column = "curricular_units_1st_sem_approved"
    groups = [frame.loc[frame["outcome"] == label, column].to_numpy() for label in OUTCOME_ORDER]
    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    axes.boxplot(groups, tick_labels=OUTCOME_ORDER, showfliers=False)
    axes.set_ylabel("approved courses, 1st semester")
    axes.set_title("1st-semester approved courses by outcome")
    figure.tight_layout()
    path = FIG_DIR / "approved_by_outcome.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_raw()
    for path in (save_outcome_balance(frame), save_approved_spread(frame)):
        print("wrote", path)


if __name__ == "__main__":
    main()
