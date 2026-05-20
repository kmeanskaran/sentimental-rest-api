import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_api.config import CONFIG_FILE
from sentiment_api.train import (
    INFO_FILE,
    METRICS_FILE,
    MODEL_FILE,
    PREDICTIONS_FILE,
    SAMPLE_PREDICTION_COUNT,
    TEST_FILE,
    TRAIN_FILE,
    predict_batch,
    save_training_outputs,
    train,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the LSTM sentiment model, evaluate it, and save JSON artifacts.",
    )
    parser.add_argument("--train-file", default=str(TRAIN_FILE))
    parser.add_argument("--test-file", default=str(TEST_FILE))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_PREDICTION_COUNT)
    return parser


def main():
    args = build_parser().parse_args()
    epochs = args.epochs if args.epochs is not None else None
    print("Running train_and_evaluate.py")
    print(f"Config file: {CONFIG_FILE}")
    print(f"Train file: {args.train_file}")
    print(f"Test file: {args.test_file}")
    print(f"Epochs: {epochs if epochs is not None else 'from config.yml'}")
    print(f"Sample prediction count: {args.sample_count}")

    train_kwargs = {
        "train_file": args.train_file,
        "test_file": args.test_file,
    }
    if epochs is not None:
        train_kwargs["epochs"] = epochs

    result = train(**train_kwargs)

    sample_examples = result["examples"]["test_binary_only"][: args.sample_count]
    print(f"Generating {len(sample_examples)} sample predictions")
    sample_predictions = predict_batch(
        model=result["model"],
        texts=[row["text"] for row in sample_examples],
        vocab=result["metadata"]["vocab"],
        max_sequence_length=result["metadata"]["max_sequence_length"],
        device=result["device"],
    )

    for prediction, row in zip(sample_predictions, sample_examples):
        prediction["true_label"] = row["label"]
        prediction["true_sentiment"] = "positive" if row["label"] == 1 else "negative"

    save_training_outputs(result["metrics"], sample_predictions)
    print(f"Saved metrics to: {METRICS_FILE}")
    print(f"Saved sample predictions to: {PREDICTIONS_FILE}")

    summary = {
        "model_file": str(MODEL_FILE),
        "metadata_file": str(INFO_FILE),
        "metrics_file": str(METRICS_FILE),
        "sample_predictions_file": str(PREDICTIONS_FILE),
        "metrics": result["metrics"],
        "sample_predictions_saved": len(sample_predictions),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
