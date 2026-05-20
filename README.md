# Twitter Sentiment Analysis API

This repository contains an end-to-end submission for the Avoma NLP task: model training, evaluation artifacts, and a FastAPI inference service.

## Important Files

```text
.
├── app.py
├── config.yml
├── requirements.txt
├── analysis.ipynb
├── scripts/
│   └── train_and_evaluate.py
├── sentiment_api/
│   ├── config.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── service.py
│   └── train.py
├── trainingandtestdata/
└── artifacts/
```

## What Each Part Does

- `app.py`: FastAPI app exposing `POST /evaluate` and `GET /health`
- `config.yml`: dataset paths, artifact path, and training hyperparameters
- `analysis.ipynb`: EDA and problem framing
- `scripts/train_and_evaluate.py`: entry point for training, evaluation, and artifact generation
- `sentiment_api/`: reusable package for preprocessing, model definition, training, and inference loading
- `trainingandtestdata/`: source CSV files
- `artifacts/`: trained model and generated JSON outputs

## Config Guide

Edit [config.yml](/Users/karan/Documents/Karan-Shingde-avoma-nlp-task/config.yml:1) before training if you want to change parameters.

Current sections:

- `paths.train_file`: training CSV path
- `paths.test_file`: manual test CSV path
- `paths.artifact_dir`: output directory for model and JSON artifacts
- `training.vocab_limit`: max vocabulary size
- `training.val_size`: validation split fraction
- `training.seed`: random seed
- `training.embed_size`: embedding dimension
- `training.hidden_size`: LSTM hidden size
- `training.lstm_layers`: number of LSTM layers
- `training.dropout`: dropout rate
- `training.batch_size`: training batch size
- `training.learning_rate`: optimizer learning rate
- `training.weight_decay`: optimizer weight decay
- `training.epochs`: epoch count
- `training.sample_prediction_count`: number of sample predictions to save

## Run Sequence

### 1. Set up the environment

Use the helper script:

```bash
sh set_uv.sh
source .venv/bin/activate
```

Or do it manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Review or update config

Open `config.yml` and adjust any paths or training parameters you want to change.

Important:

- `training.epochs` controls total epochs
- `training.batch_size` controls batch size
- `training.learning_rate` controls optimizer step size
- `training.sample_prediction_count` controls how many sample predictions are written to JSON

### 3. Train and evaluate

Run:

```bash
python scripts/train_and_evaluate.py
```

To override the configured epoch count for one run:

```bash
python scripts/train_and_evaluate.py --epochs 5
```

What it does:

1. Loads `config.yml`
2. Loads the train and test CSV files
3. Builds the train and validation split
4. Trains the LSTM model
5. Prints each epoch as `Epoch X/Y`
6. Saves the best model checkpoint
7. Evaluates on validation and filtered test data
8. Saves sample predictions to JSON

Artifacts written:

- `artifacts/best_lstm_model.pt`
- `artifacts/model_metadata.json`
- `artifacts/evaluation_metrics.json`
- `artifacts/sample_predictions.json`

### 4. Start the API

Run:

```bash
uvicorn app:app --reload
```

The API will be available at:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### 5. Call the API

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"sentence":"I love this update"}'
```

Example response shape:

```json
{
  "text": "I love this update",
  "clean_text": "i love this update",
  "predicted_label": 1,
  "predicted_sentiment": "positive",
  "probability_positive": 0.91,
  "probability_negative": 0.09
}
```

### 6. Open the notebook

If you want to review the EDA notebook:

```bash
jupyter notebook analysis.ipynb
```

## Notes

- The model is intentionally a binary classifier because the training file contains only labels `0` and `4`.
- Neutral rows with label `2` in the manual test set are excluded from scored evaluation.
- The notebook is EDA-focused; the actual training implementation lives in Python code.
