import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "ted_talks_en.csv")


def load_ted_data(csv_path=DATA_PATH, subset_size=50):
    """
    Load only a small subset to save costs.
    subset_size: number of TED talks to load
    """
    df = pd.read_csv(csv_path)
    return df.head(subset_size)


def chunk_text(text, chunk_size=1024, overlap_ratio=0.2):
    """
    Divide text into chunks for embedding.
    chunk_size: tokens per chunk
    overlap_ratio: fraction of overlap between chunks
    """
    tokens_per_chunk = chunk_size
    overlap = int(tokens_per_chunk * overlap_ratio)

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + tokens_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += tokens_per_chunk - overlap
    return chunks
