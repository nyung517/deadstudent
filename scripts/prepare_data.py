from collections import Counter

from deadstudent.data import load_raw, save_splits, split_frame
from deadstudent.features import build_design, feature_columns


def main() -> None:
    frame = load_raw()
    print(f"raw shape: {frame.shape[0]} rows x {frame.shape[1] - 2} features")
    print("outcome counts:", dict(Counter(frame["outcome"])))
    print(f"positive rate (Dropout): {frame['target'].mean():.4f}")
    print("missing cells:", int(frame.isna().sum().sum()))

    split = split_frame(frame)
    save_splits(split)
    for name in ("train", "valid", "test"):
        part = getattr(split, name)
        print(f"{name}: n={len(part)} positives={int(part['target'].sum())} rate={part['target'].mean():.4f}")

    used = feature_columns(split.train)
    print("feature columns used:", len(used))
    second_sem = [column for column in split.train.columns if column.startswith("curricular_units_2nd_sem_")]
    print("second-semester columns dropped:", len(second_sem))

    design = build_design(split)
    print("design matrix:", design.X_train.shape, design.X_valid.shape, design.X_test.shape)


if __name__ == "__main__":
    main()
