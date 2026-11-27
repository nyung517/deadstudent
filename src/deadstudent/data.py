from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/data.csv")
PROCESSED_DIR = Path("data/processed")
TARGET_COL = "target"
OUTCOME_COL = "outcome"
SECOND_SEM_PREFIX = "curricular_units_2nd_sem_"

POSITIVE_LABEL = "Dropout"
OUTCOME_TO_TARGET = {"Dropout": 1, "Enrolled": 0, "Graduate": 0}

TEST_SIZE = 0.2
VALID_SIZE = 0.2
SEED = 42


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for name in frame.columns:
        key = name.replace("\ufeff", "").strip().lower()
        key = key.replace(" (", "_").replace(")", "").replace(" ", "_").replace("'", "").replace("/", "_")
        mapping[name] = key
    return frame.rename(columns=mapping)


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    frame = normalise_columns(frame)
    frame[OUTCOME_COL] = frame["target"].astype(str).str.strip()
    frame[TARGET_COL] = frame[OUTCOME_COL].map(OUTCOME_TO_TARGET).astype("int64")
    return frame


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_frame(frame: pd.DataFrame) -> Split:
    train, rest = train_test_split(
        frame, test_size=VALID_SIZE + TEST_SIZE, stratify=frame[TARGET_COL], random_state=SEED
    )
    valid, test = train_test_split(
        rest, test_size=0.5, stratify=rest[TARGET_COL], random_state=SEED
    )
    return Split(
        train=train.sort_index().reset_index(drop=True),
        valid=valid.sort_index().reset_index(drop=True),
        test=test.sort_index().reset_index(drop=True),
    )


def save_splits(split: Split, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "valid", "test"):
        part: pd.DataFrame = getattr(split, name)
        part.to_csv(out_dir / f"{name}.csv", index=False)
