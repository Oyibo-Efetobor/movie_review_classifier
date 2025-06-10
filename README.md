# Movie Review Classifier (LSTM + GloVe)

This project classifies movie reviews as positive or negative using an LSTM neural network and pre-trained GloVe word embeddings.

## Features
- Binary sentiment classification (positive/negative)
- LSTM-based neural network
- Utilizes pre-trained GloVe word embeddings
- End-to-end workflow: data preprocessing, model training, evaluation

## Project Structure
- `data/` — Place your dataset here (see below)
- `notebooks/` — Jupyter notebooks for exploration and prototyping
- `src/` — Source code for data processing, model, and training scripts
- `glove/` — Downloaded GloVe embeddings

## Setup Instructions

### 1. Clone the repository
```
git clone <repo-url>
cd movie_review_classifier
```

### 2. Create and activate a virtual environment
```
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Download GloVe Embeddings
Download GloVe (e.g., `glove.6B.100d.txt`) from [GloVe website](https://nlp.stanford.edu/projects/glove/) and place it in the `glove/` folder.

### 5. Prepare the Dataset
- Place your movie review dataset (CSV with `review` and `sentiment` columns) in the `data/` folder.
- Example: [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

### 6. Train the Model
```
python src/train.py --data_path data/your_dataset.csv --glove_path glove/glove.6B.100d.txt
```

### 7. Evaluate the Model
```
python src/evaluate.py --model_path models/best_model.pth --data_path data/your_test_data.csv
```

## File Overview
- `src/data_preprocessing.py` — Text cleaning, tokenization, and dataset preparation
- `src/model.py` — LSTM model definition
- `src/train.py` — Training loop and model saving
- `src/evaluate.py` — Model evaluation and metrics
- `requirements.txt` — Python dependencies

## Example Usage
```
python src/train.py --data_path data/imdb_train.csv --glove_path glove/glove.6B.100d.txt
python src/evaluate.py --model_path models/best_model.pth --data_path data/imdb_test.csv
```

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- scikit-learn

## License
MIT
