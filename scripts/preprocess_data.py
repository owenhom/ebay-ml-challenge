# scripts/preprocess_data.py

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Paths
RAW_TSV = "data/raw/tagged_titles_train.tsv"
PROCESSED_DIR = "data/processed"
TRAIN_OUT = os.path.join(PROCESSED_DIR, "train.json")
VAL_OUT   = os.path.join(PROCESSED_DIR, "val.json")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# 2. Load TSV with no NA interpretation (empty string stays empty)
df = pd.read_csv(
    RAW_TSV,
    sep="\t",
    header=0,
    keep_default_na=False,
    na_values=None,
    names=["record_id","category","title","token","tag"]
)

# 3. Group by record
records = []
for rec_id, group in df.groupby("record_id", sort=False):
    group = group.reset_index(drop=True)
    category = int(group.loc[0, "category"])
    title = group.loc[0, "title"]
    tokens = []
    labels = []
    prev_tag = None

    # Optionally prepend category as a token
    cat_tok = f"[CAT{category}]"
    tokens.append(cat_tok)
    labels.append("O")

    for _, row in group.iterrows():
        tok, tag = row["token"], row["tag"]

        # Determine label
        if tag == "":  # continuation
            if prev_tag and prev_tag != "O":
                label = "I-" + prev_tag
            else:
                label = "O"
        else:
            if tag == "O":
                label = "O"
            else:
                label = "B-" + tag
        prev_tag = tag if tag != "" else prev_tag

        tokens.append(tok)
        labels.append(label)

    records.append({
        "record_id": rec_id,
        "category": category,
        "tokens": tokens,
        "labels": labels
    })

# 4. 85/15 split by record_id
train_rec, val_rec = train_test_split(
    records,
    test_size=0.15,
    random_state=42
)

# 5. Save JSONL files
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

save_jsonl(train_rec, TRAIN_OUT)
save_jsonl(val_rec, VAL_OUT)

print(f"✅ Preprocessed {len(train_rec)} train and {len(val_rec)} val records.")
print(f"Files written:\n - {TRAIN_OUT}\n - {VAL_OUT}")
