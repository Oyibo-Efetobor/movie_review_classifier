import argparse
import torch
from data_preprocessing import (set_seed, load_dataset, build_vocab, ReviewDataset)
from model import LSTMClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report


def evaluate(args):
    set_seed()
    X_train, X_val, y_train, y_val = load_dataset(args.data_path, test_size=1.0)
    vocab = build_vocab(X_train)
    max_len = 200
    test_dataset = ReviewDataset(X_train, y_train, vocab, max_len)
    test_loader = DataLoader(test_dataset, batch_size=64)

    embedding_dim = 100
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        embedding_weights=None,
        pad_idx=vocab['<PAD>']
    )
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds += (outputs > 0.5).int().tolist()
            targets += labels.int().tolist()
    acc = accuracy_score(targets, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(targets, preds, target_names=['negative', 'positive']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the test data CSV')
    args = parser.parse_args()
    evaluate(args)
