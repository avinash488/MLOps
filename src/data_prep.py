from datasets import load_dataset
from transformers import AutoTokenizer

# Load SST-2 — downloads automatically on first run (~7MB)
dataset = load_dataset("sst2")

# DistilBERT tokenizer — downloads on first run (~250MB)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128       # 128 is enough for short reviews, keeps training fast
    )

tokenized = dataset.map(tokenize, batched=True)
tokenized.save_to_disk("data/processed")

print("Done. Splits:", tokenized)