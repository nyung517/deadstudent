# deadstudent

Class project. Rank students by dropout risk after the first semester so a fixed number of
advising slots (K) can be filled before the second semester starts.

## Hard rules

- Never use `curricular_units_2nd_sem_*`. Those columns are recorded after the prediction
  moment, so they leak the label. `src/deadstudent/features.py` drops them by default.
- Fit every preprocessing statistic on the training part only. `build_design` fits the
  encoder and scaler on `train` and transforms the other parts with it.
- Do not change the split. It is stratified on `target`, `random_state=42`, 60/20/20.
  Changing the seed invalidates the numbers already in the report.
- `test` is scored once, at the end of the project, and never used for model selection.
- Predictions are ranked by `predict_proba(...)[:, 1]`. The budget metric is precision at
  the top K rows, with K fixed as a fraction of the evaluation set (K = 300 / 4424).
- Report numbers must come from script output. Never estimate a score by hand.

## Written by the student, not by the agent

The problem statement and cost asymmetry, the human rule, the pre-run predictions in
`ai_log.md`, the final model choice, the two false-case explanations, and the reflection.
Draft them only when asked, and leave the wording to the student.

## Layout

```
src/deadstudent/data.py      load, normalise columns, binary target, split
src/deadstudent/features.py  nominal/numeric groups, design matrix
scripts/prepare_data.py      writes data/processed/{train,valid,test}.csv
scripts/make_eda.py          writes artifacts/figures/*.png
```

## Commands

```
uv sync
uv run python scripts/prepare_data.py
uv run python scripts/make_eda.py
```

## Conventions

- English comments and English script output.
- snake_case module and script names, underscore infix for tool scripts.
- No PCA. Feature interpretation is done with coefficients and grouped importances.
