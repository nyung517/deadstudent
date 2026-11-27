"""Pull the most informative false positive and false negative from the test part."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from deadstudent.data import RAW_PATH, SEED, load_raw, split_frame
from deadstudent.features import build_design

K_TOTAL = 300
COLUMNS = [
    "outcome",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "tuition_fees_up_to_date",
    "debtor",
    "scholarship_holder",
    "age_at_enrollment",
    "course",
    "admission_grade",
]


def main() -> None:
    frame = load_raw(RAW_PATH)
    split = split_frame(frame)
    design = build_design(split)
    model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, random_state=SEED)
    model.fit(design.X_train, design.y_train)
    probability = model.predict_proba(design.X_test)[:, 1]
    test = split.test.reset_index(drop=True).assign(probability=probability)
    test["rank"] = test["probability"].rank(ascending=False, method="first").astype(int)
    k = round(K_TOTAL / 4424 * len(test))

    print("test rows=%d positives=%d K=%d" % (len(test), int(test["target"].sum()), k))
    print("flagged at 0.5 = %d, inside top %d = %d" % (int((test["probability"] >= 0.5).sum()), k, int((test["rank"] <= k).sum())))

    false_positive = test[test["target"] == 0].sort_values("probability", ascending=False).head(1)
    missed = test[test["target"] == 1].sort_values("probability").head(1)
    outside_k = np.logical_and(test["target"].to_numpy() == 1, test["rank"].to_numpy() > k)
    near_miss = test[outside_k].sort_values("rank").head(1)

    for label, rows in (
        ("FALSE POSITIVE (highest score, stayed)", false_positive),
        ("FALSE NEGATIVE (lowest score, dropped out)", missed),
        ("NEAR MISS (first dropout outside K)", near_miss),
    ):
        print("")
        print("==", label)
        print(rows[COLUMNS + ["probability", "rank"]].to_string(index=False))

    print("")
    print("== medians on test")
    print(test.groupby("outcome")[["curricular_units_1st_sem_approved", "curricular_units_1st_sem_grade"]].median())


if __name__ == "__main__":
    main()
