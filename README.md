# Twitter Sentiment Analysis API

This repository contains an end-to-end submission for the Avoma NLP task: model training, evaluation artifacts, and a FastAPI inference service.

## Important Files

```text
.
├── app.py
├── config.yml
├── requirements.txt
├── set_uv.sh
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

- `app.py`: FastAPI app exposing `POST /evaluate`, `POST /evaluate/batch`, and `GET /health`
- `config.yml`: dataset paths, artifact path, and training hyperparameters
- `set_uv.sh`: helper setup script that prepares a local `.venv` using `uv` and installs dependencies from `requirements.txt`
- `analysis.ipynb`: EDA and problem framing
- `scripts/train_and_evaluate.py`: entry point for training, evaluation, and artifact generation
- `sentiment_api/`: reusable package for preprocessing, model definition, training, and inference loading
- `trainingandtestdata/`: source CSV files
- `artifacts/`: trained model and generated JSON outputs

## Config Guide

Edit [config.yml](/Users/karan/Documents/sentimental-rest-api/config.yml:1) before training if you want to change parameters.

Current sections:

- `paths.train_file`: training CSV path
- `paths.test_file`: manual test CSV path
- `paths.artifact_dir`: output directory for model and JSON artifacts
- `training.vocab_limit`: max vocabulary size
- `training.deduplicate_text`: remove duplicate cleaned tweets before splitting and drop conflicting-label duplicates
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

What `set_uv.sh` does:

- Sets project-local `uv` cache directories under `.cache/uv`
- Reuses an existing `.venv` if present, or creates one if it does not exist
- Installs `uv` first if it is not already available on your machine
- Installs all packages from `requirements.txt` into `.venv`

Use this script when you want a repeatable local setup without manually creating the virtual environment and installing dependencies step by step.

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
3. Deduplicates exact cleaned tweets before splitting when enabled
4. Builds the train and validation split
5. Trains the LSTM model
6. Prints each epoch as `Epoch X/Y`
7. Saves the best model checkpoint
8. Evaluates on validation and filtered test data
9. Saves sample predictions to JSON

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

Batch request example:

```bash
curl -X POST "http://127.0.0.1:8000/evaluate/batch" \
  -H "Content-Type: application/json" \
  -d '{"sentences":["I love this update","This is the worst release yet"]}'
```

Batch response shape:

```json
{
  "predictions": [
    {
      "text": "I love this update",
      "clean_text": "i love this update",
      "predicted_label": 1,
      "predicted_sentiment": "positive",
      "probability_positive": 0.91,
      "probability_negative": 0.09
    },
    {
      "text": "This is the worst release yet",
      "clean_text": "this is the worst release yet",
      "predicted_label": 0,
      "predicted_sentiment": "negative",
      "probability_positive": 0.08,
      "probability_negative": 0.92
    }
  ]
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
- Exact duplicate cleaned tweets are removed before the train/validation split by default to reduce leakage; conflicting-label duplicates are dropped.
- The notebook is EDA-focused; the actual training implementation lives in Python code.
