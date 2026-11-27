"""Recompute the headline numbers without scikit-learn, so they can be hand-checked.

Every block below prints the counts it used, not just the score.  Deliberately
sklearn-free: pandas + numpy only, plus plain Python loops for the rule and for
precision@K.  That makes this script an independent check on the numbers in the
report rather than a copy of ``run_models.py``.

What it checks:
  1. positive rate on the valid part              -> positives / n
  2. human rule precision and recall              -> explicit loop over the 885 rows
  3. precision@K from artifacts/valid_scores.csv  -> top-K hits / K (skipped if absent)

Run:
    python scripts/verify_numbers.py
    python scripts/verify_numbers.py --scores artifacts/valid_scores.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VALID_PATH = Path("data/processed/valid.csv")
SCORES_PATH = Path("artifacts/valid_scores.csv")
TARGET_COL = "target"

BUDGET_TOTAL = 300
DATASET_ROWS = 4424
MIN_APPROVED = 6
APPROVED_COL = "curricular_units_1st_sem_approved"
TUITION_COL = "tuition_fees_up_to_date"


def load_valid(path: Path = VALID_PATH) -> pd.DataFrame:
    """Read the valid split written by scripts/prepare_data.py."""
    if not path.exists():
        raise SystemExit(f"missing {path}: run scripts/prepare_data.py first")
    return pd.read_csv(path)


def report_positive_rate(frame: pd.DataFrame) -> float:
    """positives / n on the valid part."""
    labels = frame[TARGET_COL].to_numpy()
    n = len(labels)
    positives = int(labels.sum())
    rate = positives / n
    print("[1] positive rate on valid (target = Dropout)")
    print(f"    n = {n}, positives = {positives}")
    print(f"    rate = positives / n = {positives} / {n} = {rate:.4f}")
    return rate


def human_rule_counts(frame: pd.DataFrame) -> tuple[int, int, int, int, int]:
    """Run the human rule row by row and return tp, fp, fn, tn, fired."""
    tp = fp = fn = tn = fired = 0
    for approved, paid, label in zip(
        frame[APPROVED_COL].to_numpy(),
        frame[TUITION_COL].to_numpy(),
        frame[TARGET_COL].to_numpy(),
    ):
        predicted = 1 if (approved < MIN_APPROVED or paid == 0) else 0
        fired += predicted
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn, fired


def report_human_rule(frame: pd.DataFrame) -> tuple[float, float]:
    """precision = tp / (tp + fp), recall = tp / (tp + fn)."""
    tp, fp, fn, tn, fired = human_rule_counts(frame)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print("")
    print("[2] human rule: flag if (1st_sem_approved < 6) OR (tuition_fees_up_to_date == 0)")
    print(f"    flagged rows = {fired} of {len(frame)}")
    print(f"    tp = {tp} (flagged and Dropout)      fp = {fp} (flagged but not Dropout)")
    print(f"    fn = {fn} (not flagged but Dropout)  tn = {tn} (not flagged, not Dropout)")
    print(f"    check: tp + fp + fn + tn = {tp + fp + fn + tn} (must equal {len(frame)})")
    print(f"    precision = tp / (tp + fp) = {tp} / ({tp} + {fp}) = {precision:.4f}")
    print(f"    recall    = tp / (tp + fn) = {tp} / ({tp} + {fn}) = {recall:.4f}")
    return precision, recall


def budget_k(n: int) -> int:
    """Same rule as run_models.py: K = round(300 / 4424 * n), at least 1."""
    return max(1, round(BUDGET_TOTAL / DATASET_ROWS * n))


def precision_at_k_by_loop(scores: np.ndarray, labels: np.ndarray, k: int) -> tuple[int, float]:
    """Rank rows by descending score and count positives in the top k."""
    order = sorted(range(len(scores)), key=lambda position: (-scores[position], position))
    hits = 0
    for position in order[:k]:
        if labels[position] == 1:
            hits += 1
    return hits, hits / k


def report_precision_at_k(frame: pd.DataFrame, path: Path = SCORES_PATH) -> None:
    """precision@K for every probability column in the scores file."""
    print("")
    print("[3] precision@K from the saved valid predictions")
    if not path.exists():
        print(f"    missing: {path} does not exist yet.")
        print("    precision@K needs one probability column per model, written by")
        print("    scripts/run_models.py when it runs without --smoke.")
        print("    Write your own predictions first, then re-run this script.")
        return

    scores_frame = pd.read_csv(path, index_col=0)
    labels_by_row = pd.Series(frame[TARGET_COL].to_numpy(), index=frame.index)
    unknown = [index for index in scores_frame.index if index not in labels_by_row.index]
    if unknown:
        print(f"    warning: {len(unknown)} score rows are not in {VALID_PATH} (first: {unknown[0]})")
        return
    labels = labels_by_row.loc[scores_frame.index].to_numpy()

    k = budget_k(len(labels))
    positives = int(labels.sum())
    print(f"    rows = {len(labels)}, positives = {positives}")
    print(f"    K = round({BUDGET_TOTAL} / {DATASET_ROWS} * {len(labels)}) = {k}")
    print(f"    {'model':<26} {'hits@K':>6} {'K':>4} {'precision@K':>12} {'base rate':>10}")
    for column in scores_frame.columns:
        scores = scores_frame[column].to_numpy(dtype=float)
        hits, value = precision_at_k_by_loop(scores, labels, k)
        print(f"    {column:<26} {hits:>6} {k:>4} {value:>12.4f} {positives / len(labels):>10.4f}")
    print(f"    precision@K = hits in top K / K; base rate = {positives} / {len(labels)} = "
          f"{positives / len(labels):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="sklearn-free recheck of the report numbers")
    parser.add_argument("--scores", type=Path, default=SCORES_PATH, help="saved valid probabilities")
    args = parser.parse_args()

    frame = load_valid()
    report_positive_rate(frame)
    report_human_rule(frame)
    report_precision_at_k(frame, args.scores)


if __name__ == "__main__":
    main()
