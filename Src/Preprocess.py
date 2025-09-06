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
