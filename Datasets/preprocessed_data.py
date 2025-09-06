from datasets import load_dataset
import re

def clean_text(text: str) -> str:
    """
    Cleans tweet text by removing URLs, mentions, hashtags, punctuation, and extra spaces.
    """
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)          # remove URLs
    text = re.sub(r"@\w+", " ", text)             # remove mentions
    text = re.sub(r"#", " ", text)                # remove hashtags symbol
    text = re.sub(r"[^a-z\s]", " ", text)         # keep only letters
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    return text


# Load dataset
dataset = load_dataset("melisekm/natural-disasters-from-social-media")

# Apply preprocessing
def preprocess_batch(batch):
    batch["text"] = [clean_text(t) for t in batch["text"]]
    return batch

cleaned_dataset = dataset.map(preprocess_batch, batched=True)

# Remove unnecessary columns
cleaned_dataset = cleaned_dataset.remove_columns(
    ["tweet_id", "SOURCE_FILE", "filename"]
)

print(cleaned_dataset)

# Optionally save preprocessed dataset
cleaned_dataset.save_to_disk("Datasets/processed")
