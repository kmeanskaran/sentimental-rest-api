import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sentiment_api.config import load_config, resolve_repo_path
from sentiment_api.model import SentimentLSTM
from sentiment_api.preprocessing import (
    PAD_INDEX,
    build_feature_tensor,
    build_vocab,
    load_binary_examples,
    percentile_sequence_length,
    stratified_split,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config()
PATH_CONFIG = CONFIG["paths"]
TRAINING_CONFIG = CONFIG["training"]

TRAIN_FILE = resolve_repo_path(PATH_CONFIG["train_file"])
TEST_FILE = resolve_repo_path(PATH_CONFIG["test_file"])

ARTIFACT_DIR = resolve_repo_path(PATH_CONFIG["artifact_dir"])
MODEL_FILE = ARTIFACT_DIR / "best_lstm_model.pt"
INFO_FILE = ARTIFACT_DIR / "model_metadata.json"
METRICS_FILE = ARTIFACT_DIR / "evaluation_metrics.json"
PREDICTIONS_FILE = ARTIFACT_DIR / "sample_predictions.json"

VOCAB_LIMIT = TRAINING_CONFIG["vocab_limit"]
VAL_SIZE = TRAINING_CONFIG["val_size"]
SEED = TRAINING_CONFIG["seed"]
EMBED_SIZE = TRAINING_CONFIG["embed_size"]
HIDDEN_SIZE = TRAINING_CONFIG["hidden_size"]
LSTM_LAYERS = TRAINING_CONFIG["lstm_layers"]
DROPOUT = TRAINING_CONFIG["dropout"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
WEIGHT_DECAY = TRAINING_CONFIG["weight_decay"]
EPOCHS = TRAINING_CONFIG["epochs"]
SAMPLE_PREDICTION_COUNT = TRAINING_CONFIG["sample_prediction_count"]

LABEL_NAMES = {
    0: "negative",
    1: "positive",
}


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_model(vocab_size, device):
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBED_SIZE,
        hidden_dim=HIDDEN_SIZE,
        num_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        pad_index=PAD_INDEX,
    )
    return model.to(device)


def prepare_datasets(
    train_file=TRAIN_FILE,
    test_file=TEST_FILE,
    val_size=VAL_SIZE,
    vocab_limit=VOCAB_LIMIT,
    seed=SEED,
):
    train_file = Path(train_file)
    test_file = Path(test_file)

    print(f"Loading training data from: {train_file}")
    print(f"Loading test data from: {test_file}")

    all_train_examples = load_binary_examples(train_file)
    test_examples = load_binary_examples(test_file)

    train_examples, val_examples = stratified_split(
        all_train_examples,
        val_fraction=val_size,
        seed=seed,
    )

    vocab = build_vocab(train_examples, vocab_limit)
    max_sequence_length = percentile_sequence_length(
        [row["text"] for row in all_train_examples],
        percentile=0.95,
    )

    print(f"Loaded {len(all_train_examples)} training examples")
    print(f"Loaded {len(test_examples)} filtered test examples")
    print(f"Training split size: {len(train_examples)}")
    print(f"Validation split size: {len(val_examples)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Max sequence length: {max_sequence_length}")

    datasets = {
        "train": TensorDataset(
            build_feature_tensor([row["text"] for row in train_examples], vocab, max_sequence_length),
            torch.tensor([row["label"] for row in train_examples], dtype=torch.float32),
        ),
        "validation": TensorDataset(
            build_feature_tensor([row["text"] for row in val_examples], vocab, max_sequence_length),
            torch.tensor([row["label"] for row in val_examples], dtype=torch.float32),
        ),
        "test_binary_only": TensorDataset(
            build_feature_tensor([row["text"] for row in test_examples], vocab, max_sequence_length),
            torch.tensor([row["label"] for row in test_examples], dtype=torch.float32),
        ),
    }

    return {
        "vocab": vocab,
        "max_sequence_length": max_sequence_length,
        "datasets": datasets,
        "examples": {
            "train": train_examples,
            "validation": val_examples,
            "test_binary_only": test_examples,
        },
    }


def make_data_loaders(datasets, batch_size=BATCH_SIZE):
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False),
        "test_binary_only": DataLoader(datasets["test_binary_only"], batch_size=batch_size, shuffle=False),
    }


def get_scores(logits, labels):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

    accuracy = (predictions == labels).float().mean().item()
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * features.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    scores = get_scores(all_logits, all_labels)
    scores["loss"] = total_loss / len(data_loader.dataset)
    return scores


def train(
    train_file=TRAIN_FILE,
    test_file=TEST_FILE,
    epochs=EPOCHS,
):
    set_seed(SEED)
    device = get_device()
    print("Starting training run")
    print(f"Using device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Seed: {SEED}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    prepared = prepare_datasets(train_file=train_file, test_file=test_file)
    data_loaders = make_data_loaders(prepared["datasets"])

    model = make_model(len(prepared["vocab"]), device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        model.train()
        train_loss_sum = 0.0
        train_logits = []
        train_labels = []

        for features, labels in data_loaders["train"]:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * features.size(0)
            train_logits.append(logits.detach().cpu())
            train_labels.append(labels.detach().cpu())

        train_logits = torch.cat(train_logits)
        train_labels = torch.cat(train_labels)
        train_scores = get_scores(train_logits, train_labels)
        train_scores["loss"] = train_loss_sum / len(data_loaders["train"].dataset)

        validation_scores = evaluate_model(
            model,
            data_loaders["validation"],
            loss_fn,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_scores,
                "validation": validation_scores,
            }
        )

        print(
            "  "
            f"train_loss={train_scores['loss']:.4f} "
            f"train_acc={train_scores['accuracy']:.4f} "
            f"val_loss={validation_scores['loss']:.4f} "
            f"val_acc={validation_scores['accuracy']:.4f} "
            f"val_f1={validation_scores['f1']:.4f}"
        )

        if validation_scores["f1"] > best_val_f1:
            best_val_f1 = validation_scores["f1"]
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"  Saved new best model to: {MODEL_FILE}")

    metadata = {
        "vocab": prepared["vocab"],
        "max_sequence_length": prepared["max_sequence_length"],
    }
    INFO_FILE.write_text(json.dumps(metadata))
    print(f"Saved model metadata to: {INFO_FILE}")

    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    metrics = {
        "validation": evaluate_model(model, data_loaders["validation"], loss_fn, device),
        "test_binary_only": evaluate_model(model, data_loaders["test_binary_only"], loss_fn, device),
    }
    print("Final evaluation metrics:")
    print(f"  validation={metrics['validation']}")
    print(f"  test_binary_only={metrics['test_binary_only']}")

    return {
        "device": device,
        "model": model,
        "metadata": metadata,
        "metrics": metrics,
        "history": history,
        "examples": prepared["examples"],
    }


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_training_outputs(metrics, sample_predictions):
    save_json(METRICS_FILE, metrics)
    save_json(PREDICTIONS_FILE, sample_predictions)


def evaluate(
    model,
    text,
    vocab,
    max_sequence_length,
    device,
):
    from sentiment_api.preprocessing import clean_tweet, encode_text

    clean_text = clean_tweet(text)
    ids = encode_text(clean_text, vocab, max_sequence_length)
    features = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        score = model(features)
        probability_positive = torch.sigmoid(score).item()

    predicted_label = 1 if probability_positive >= 0.5 else 0
    return {
        "text": text,
        "clean_text": clean_text,
        "predicted_label": predicted_label,
        "predicted_sentiment": LABEL_NAMES[predicted_label],
        "probability_positive": probability_positive,
        "probability_negative": 1 - probability_positive,
    }


def predict_batch(
    model,
    texts,
    vocab,
    max_sequence_length,
    device,
):
    return [
        evaluate(
            model=model,
            text=text,
            vocab=vocab,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        for text in texts
    ]
