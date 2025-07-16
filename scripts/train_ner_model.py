import os
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import precision_score, recall_score, f1_score

# ----- Configuration -----
MODEL_CHECKPOINT = "bert-base-multilingual-cased"
TRAIN_FILE = "data/processed/train.json"
VAL_FILE   = "data/processed/val.json"
OUTPUT_DIR = "models/ner-model"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Heartbeat Print: Start -----
print("🚀 Starting train_ner_model script", flush=True)

# ----- Load Datasets -----
datasets = load_dataset("json", data_files={
    "train": TRAIN_FILE,
    "validation": VAL_FILE
})
print(
    f"✅ Loaded datasets: train={len(datasets['train'])} examples, validation={len(datasets['validation'])} examples",
    flush=True
)

# ----- Prepare Label Mappings -----
base_labels = set()
for split in datasets:
    for labels in datasets[split]["labels"]:
        for lab in labels:
            if lab == "O":
                continue
            base_labels.add(lab.split("-", 1)[1])
label_list = ["O"] + [f"B-{b}" for b in sorted(base_labels)] + [f"I-{b}" for b in sorted(base_labels)]
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}

# ----- Load Tokenizer & Model -----
print(f"⏳ Loading tokenizer and model from checkpoint '{MODEL_CHECKPOINT}'...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id
)
print("✅ Tokenizer and model loaded.", flush=True)

# ----- Tokenization & Label Alignment -----
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    aligned_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[labels[word_idx]])
            else:
                label = labels[word_idx]
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                label_ids.append(label_to_id[label])
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

print("⏳ Tokenization & label alignment starting…", flush=True)
tokenized_datasets = datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=["record_id", "category", "tokens", "labels"]
)
print("✅ Tokenization & alignment done.", flush=True)

# ----- Data Collator -----
data_collator = DataCollatorForTokenClassification(tokenizer)

# ----- Metrics -----
def compute_metrics(p):
    preds, labs = p
    preds = np.argmax(preds, axis=2)
    true_labels, true_preds = [], []
    for pred, lab in zip(preds, labs):
        pl, ll = [], []
        for p_id, l_id in zip(pred, lab):
            if l_id == -100:
                continue
            ll.append(id_to_label[l_id])
            pl.append(id_to_label[p_id])
        true_labels.append(ll)
        true_preds.append(pl)
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds)
    }

# ----- TrainingArguments & Trainer -----
training_args = training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",            # evaluate at end of each epoch
    save_strategy="epoch",            # save at end of each epoch to match eval
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_steps=1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

# ----- Train & Save -----
print("⏳ Starting training…", flush=True)
trainer.train()
print("✅ Training complete.", flush=True)

trainer.save_model(OUTPUT_DIR)
print(f"✅ Model saved to: {OUTPUT_DIR}", flush=True)
