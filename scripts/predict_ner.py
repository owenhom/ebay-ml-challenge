import os
import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ----- Configuration -----
MODEL_DIR   = "models/ner-model"
INPUT_TSV   = "data/raw/listing_titles.tsv"
OUTPUT_DIR  = "outputs/predictions"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "quiz_preds.tsv")
CHUNK_SIZE  = 50000   # adjust if you run out of memory

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Sanity check: checkpoint exists -----
config_path = os.path.join(MODEL_DIR, "config.json")
bin_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
safetensors_path = os.path.join(MODEL_DIR, "model.safetensors")
if not os.path.isfile(config_path) or not (os.path.isfile(bin_path) or os.path.isfile(safetensors_path)):
    raise FileNotFoundError(
        f"Checkpoint folder {MODEL_DIR} is missing required files. "
        "Expected 'config.json' and at least one of 'pytorch_model.bin' or 'model.safetensors'."
    )

# ----- Load model and tokenizer -----
print(f"⏳ Loading model from {MODEL_DIR}…", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
print("✅ Model loaded.", flush=True)

# ----- Create pipeline -----
ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# ----- Inference & Write Predictions -----
print(f"⏳ Reading {INPUT_TSV} in chunks of {CHUNK_SIZE}…", flush=True)
with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as out_f:
    writer = csv.writer(out_f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
    for chunk in pd.read_csv(
        INPUT_TSV,
        sep="\t",
        header=0,
        names=["record_id", "category_id", "title"],
        dtype=str,
        keep_default_na=False,
        na_values=None,
        chunksize=CHUNK_SIZE
    ):
        for _, row in chunk.iterrows():
            rec_id = row["record_id"]
            cat_id = row["category_id"]
            text   = row["title"]
            entities = ner_pipeline(text)
            for ent in entities:
                # 'entity_group' like 'B-Tag' or direct label
                label = ent.get("entity_group", ent.get("entity"))
                if "-" in label:
                    label = label.split("-", 1)[1]
                value = ent.get("word", ent.get("entity_text"))
                writer.writerow([rec_id, cat_id, label, value])

print(f"✅ Predictions written to {OUTPUT_FILE}", flush=True)


