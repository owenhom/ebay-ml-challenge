# scripts/format_submission.py

import pandas as pd
import csv
import os

# Paths
RAW_PREDICTIONS = "outputs/predictions/quiz_preds.tsv"
SUBMISSION_FILE = "outputs/submission_quiz.tsv"

# Read raw predictions (no header)
df = pd.read_csv(
    RAW_PREDICTIONS,
    sep="\t",
    header=None,
    names=["record_id", "category_id", "aspect_name", "aspect_value"],
    dtype=str,
    keep_default_na=False,
    na_values=None
)

# Validate
required = ["record_id", "category_id", "aspect_name", "aspect_value"]
if not all(col in df.columns for col in required):
    raise ValueError(f"Raw predictions must have columns {required}")

# Write submission TSV (no header, no quoting)
os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
df.to_csv(
    SUBMISSION_FILE,
    sep="\t",
    index=False,
    header=False,
    quoting=csv.QUOTE_NONE,
    encoding="utf-8"
)

print(f"✅ Submission file created: {SUBMISSION_FILE}")
