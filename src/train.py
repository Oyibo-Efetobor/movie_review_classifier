import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import (set_seed, load_dataset, build_vocab, ReviewDataset)
from model import LSTMClassifier
import numpy as np

# Load GloVe embeddings
def load_glove_embeddings(glove_path, vocab, embedding_dim):
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[vocab[word]] = vector
    return torch.tensor(embeddings, dtype=torch.float)

def train(args):
    set_seed()
    X_train, X_val, y_train, y_val = load_dataset(args.data_path)
    vocab = build_vocab(X_train)
    max_len = 200
    train_dataset = ReviewDataset(X_train, y_train, vocab, max_len)
    val_dataset = ReviewDataset(X_val, y_val, vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    embedding_dim = 100
    embedding_weights = load_glove_embeddings(args.glove_path, vocab, embedding_dim)
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        embedding_weights=embedding_weights,
        pad_idx=vocab['<PAD>']
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    best_val_loss = float('inf')
    os.makedirs('models', exist_ok=True)
    for epoch in range(1, 11):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data CSV')
    parser.add_argument('--glove_path', type=str, required=True, help='Path to the GloVe embeddings file')
    args = parser.parse_args()
    train(args)
