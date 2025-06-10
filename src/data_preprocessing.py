import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Text cleaning and tokenization
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return text.split()

# Build vocabulary from training data
def build_vocab(texts, min_freq=2):
    freq = {}
    for text in texts:
        for token in tokenize(clean_text(text)):
            freq[token] = freq.get(token, 0) + 1
    vocab = {word: idx+2 for idx, (word, count) in enumerate(freq.items()) if count >= min_freq}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

# Encode text to sequence of indices
def encode_text(text, vocab, max_len):
    tokens = tokenize(clean_text(text))
    seq = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(seq) < max_len:
        seq += [vocab['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# Custom Dataset class
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        seq = encode_text(text, self.vocab, self.max_len)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# Load and preprocess dataset
def load_dataset(path, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    texts = df['review'].astype(str).tolist()
    labels = df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val
