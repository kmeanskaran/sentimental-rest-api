import json

import torch

from sentiment_api.model import SentimentLSTM
from sentiment_api.preprocessing import (
    PAD_INDEX,
    clean_tweet,
    encode_text,
)
from sentiment_api.train import (
    DROPOUT,
    EMBED_SIZE,
    HIDDEN_SIZE,
    INFO_FILE,
    LABEL_NAMES,
    LSTM_LAYERS,
    MODEL_FILE,
)

MODEL_PATH = MODEL_FILE
INFO_PATH = INFO_FILE


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_info():
    data = json.loads(INFO_PATH.read_text())
    vocab = {}
    for word, index in data["vocab"].items():
        vocab[word] = int(index)
    max_len = int(data.get("max_len", data.get("max_sequence_length")))
    return vocab, max_len


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


class SimpleSentimentModel:
    def __init__(self):
        self.device = get_device()
        self.vocab = None
        self.max_len = None
        self.model = None

    def load(self):
        if self.model is not None:
            return

        self.vocab, self.max_len = load_info()
        self.model = make_model(len(self.vocab), self.device)
        weights = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.eval()

    def predict(self, text):
        self.load()

        clean_text = clean_tweet(text)
        ids = encode_text(clean_text, self.vocab, self.max_len)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            score = self.model(x)
            positive = torch.sigmoid(score).item()

        label = 1 if positive >= 0.5 else 0
        return {
            "text": text,
            "clean_text": clean_text,
            "predicted_label": label,
            "predicted_sentiment": LABEL_NAMES[label],
            "probability_positive": positive,
            "probability_negative": 1 - positive,
        }

    def predict_batch(self, texts):
        return [self.predict(text) for text in texts]
