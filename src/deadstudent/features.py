from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from deadstudent.data import OUTCOME_COL, SECOND_SEM_PREFIX, TARGET_COL, Split

NOMINAL_FEATURES: tuple[str, ...] = (
    "marital_status",
    "application_mode",
    "application_order",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "nacionality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "previous_qualification_grade",
    "admission_grade",
    "age_at_enrollment",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
)

DROP_ALWAYS: tuple[str, ...] = (TARGET_COL, OUTCOME_COL)


def feature_columns(frame: pd.DataFrame, drop_second_semester: bool = True) -> list[str]:
    dropped = set(DROP_ALWAYS)
    if drop_second_semester:
        dropped |= {column for column in frame.columns if column.startswith(SECOND_SEM_PREFIX)}
    return [column for column in frame.columns if column not in dropped]


@dataclass(frozen=True)
class Design:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def build_design(split: Split, drop_second_semester: bool = True) -> Design:
    columns = feature_columns(split.train, drop_second_semester=drop_second_semester)
    nominal = [column for column in columns if column in NOMINAL_FEATURES]
    numeric = [column for column in columns if column in NUMERIC_FEATURES]

    transformer = ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    transformer.fit(split.train[columns])

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        values = transformer.transform(frame[columns])
        names = list(transformer.get_feature_names_out())
        return pd.DataFrame(values, columns=names, index=frame.index)

    return Design(
        X_train=encode(split.train),
        X_valid=encode(split.valid),
        X_test=encode(split.test),
        y_train=split.train[TARGET_COL],
        y_valid=split.valid[TARGET_COL],
        y_test=split.test[TARGET_COL],
    )
