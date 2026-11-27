import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from deadstudent.data import RAW_PATH, SEED, TARGET_COL, load_raw, split_frame
from deadstudent.features import build_design

SCOREBOARD = Path("artifacts/scoreboard.csv")
K_TOTAL = 300
HUMAN_RULE_MIN_APPROVED = 6


@dataclass(frozen=True)
class Result:
    name: str
    scores: np.ndarray
    precision: float
    recall: float
    f1: float
    auc: float
    precision_at_k: float


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    return float(np.asarray(y_true)[order].mean())


def human_rule_scores(frame: pd.DataFrame) -> np.ndarray:
    approved = frame["curricular_units_1st_sem_approved"].to_numpy()
    paid = frame["tuition_fees_up_to_date"].to_numpy()
    return ((approved < HUMAN_RULE_MIN_APPROVED) | (paid == 0)).astype(float)


def make_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, random_state=SEED
        ),
    }


def evaluate(name: str, y_true: np.ndarray, scores: np.ndarray, k: int) -> Result:
    predicted = (scores >= 0.5).astype(int)
    return Result(
        name=name,
        scores=scores,
        precision=precision_score(y_true, predicted, zero_division=0),
        recall=recall_score(y_true, predicted, zero_division=0),
        f1=f1_score(y_true, predicted, zero_division=0),
        auc=roc_auc_score(y_true, scores),
        precision_at_k=precision_at_k(y_true, scores, k),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run on the training part only")
    args = parser.parse_args()

    frame = load_raw(RAW_PATH)
    split = split_frame(frame)
    if args.smoke:
        split = type(split)(
            train=split.train.head(200).reset_index(drop=True),
            valid=split.valid.head(50).reset_index(drop=True),
            test=split.test,
        )
    design = build_design(split)
    y_valid = np.asarray(design.y_valid)
    k = max(1, round(K_TOTAL / 4424 * len(y_valid)))

    results = [
        evaluate("do_nothing", y_valid, np.zeros(len(y_valid)), k),
        evaluate("human_rule", y_valid, human_rule_scores(split.valid), k),
    ]
    for name, model in make_models().items():
        model.fit(design.X_train, design.y_train)
        results.append(evaluate(name, y_valid, model.predict_proba(design.X_valid)[:, 1], k))

    rows = [
        {
            "rank": 0,
            "model": result.name,
            "precision": round(result.precision, 4),
            "recall": round(result.recall, 4),
            "f1": round(result.f1, 4),
            "auc": round(result.auc, 4),
            "precision_at_k": round(result.precision_at_k, 4),
            "k": k,
        }
        for result in results
    ]
    table = pd.DataFrame(rows).sort_values("precision_at_k", ascending=False)
    table["rank"] = range(1, len(table) + 1)
    if not args.smoke:
        SCOREBOARD.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(SCOREBOARD, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
