import mlflow
import mlflow.pytorch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ── Config ────────────────────────────────────────────────
CHECKPOINT = "distilbert-base-uncased"
DATA_PATH  = "src/data/processed"
MODEL_OUT  = "src/models/sentiment"
# ─────────────────────────────────────────────────────────

dataset   = load_from_disk(DATA_PATH)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)

# SST-2 test split has no labels, so we only use train + validation
train_ds = dataset["train"]
val_ds   = dataset["validation"]

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir          = MODEL_OUT,
    num_train_epochs    = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    learning_rate       = 2e-5,
    eval_strategy       = "epoch",
    save_strategy       = "epoch",
    load_best_model_at_end = True,
    logging_steps       = 100,
    report_to           = "none",   # we handle logging manually via MLflow
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    processing_class       = tokenizer,
    data_collator   = DataCollatorWithPadding(tokenizer),
    compute_metrics = compute_metrics,
)

# ── MLflow run ────────────────────────────────────────────
mlflow.set_experiment("sentiment-analysis")

with mlflow.start_run(run_name="distilbert-sst2-v1"):

    mlflow.log_params({
        "model":        CHECKPOINT,
        "epochs":       training_args.num_train_epochs,
        "batch_size":   training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "max_length":   128,
        "train_samples": len(train_ds),
    })

    trainer.train()

    # Log per-epoch metrics
    for log in trainer.state.log_history:
        if "eval_accuracy" in log:
            mlflow.log_metric("val_accuracy", log["eval_accuracy"], step=int(log["epoch"]))
            mlflow.log_metric("val_loss",     log["eval_loss"],     step=int(log["epoch"]))

    # Log final accuracy
    final_acc = trainer.state.log_history[-2]["eval_accuracy"]
    mlflow.log_metric("final_val_accuracy", final_acc)

    # Save model artifact to MLflow
    mlflow.pytorch.log_model(model, "model")

    # Also save locally for the FastAPI server later
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)

    print(f"\nDone. Final val accuracy: {final_acc:.4f}")
    print(f"Model saved to: {MODEL_OUT}")