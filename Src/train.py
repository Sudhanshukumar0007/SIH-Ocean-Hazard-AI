from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

import numpy as np
import evaluate
import torch

# Check device
if torch.cuda.is_available():
    print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚡ Using CPU (no GPU detected)")

# 1. Load dataset directly from HuggingFace
dataset = load_dataset("melisekm/natural-disasters-from-social-media")

train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# 2. Load tokenizer & model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_data = train_data.map(tokenize_fn, batched=True)
val_data = val_data.map(tokenize_fn, batched=True)
test_data = test_data.map(tokenize_fn, batched=True)

# 3. Rename target → labels (required for Trainer)
train_data = train_data.rename_column("target", "labels")
val_data = val_data.rename_column("target", "labels")
test_data = test_data.rename_column("target", "labels")

# HuggingFace requires tensors
train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 4. Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. Define metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels),
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")
    }

# 6. Training args
training_args = TrainingArguments(
    output_dir="./Results",
    eval_strategy="epoch",   # <-- correct for your version
    save_strategy="epoch",
    logging_dir="./Logs",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)


# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Train
trainer.train()

# 9. Save model
model.save_pretrained("Models/bert_hazard_model")
tokenizer.save_pretrained("Models/bert_hazard_model")


