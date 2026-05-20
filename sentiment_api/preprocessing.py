import csv
import math
import random
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_INDEX = 0
UNK_INDEX = 1


def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"@\w+", " user ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9'!?., ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text):
    return text.split()


def load_binary_examples(csv_path):
    examples = []
    with csv_path.open(encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            sentiment = int(row[0])
            if sentiment not in (0, 4):
                continue
            label = 1 if sentiment == 4 else 0
            examples.append({"text": clean_tweet(row[5]), "label": label})
    return examples


def deduplicate_examples(examples, drop_conflicting_labels=True):
    grouped_examples = defaultdict(list)
    for example in examples:
        grouped_examples[str(example["text"])].append(example)

    deduplicated_examples = []
    duplicate_count = 0
    conflicting_text_count = 0

    for text, text_examples in grouped_examples.items():
        duplicate_count += max(0, len(text_examples) - 1)
        labels = {example["label"] for example in text_examples}

        if len(labels) > 1:
            conflicting_text_count += 1
            if drop_conflicting_labels:
                continue

        deduplicated_examples.append(text_examples[0])

    return deduplicated_examples, {
        "input_examples": len(examples),
        "output_examples": len(deduplicated_examples),
        "duplicate_rows_removed": duplicate_count,
        "conflicting_texts_dropped": conflicting_text_count,
    }


def percentile_sequence_length(texts, percentile):
    lengths = sorted(len(tokenize_text(text)) for text in texts)
    if not lengths:
        return 1
    rank = max(0, math.ceil(percentile * len(lengths)) - 1)
    return max(1, lengths[rank])


def stratified_split(
    examples,
    val_fraction=0.2,
    seed=42,
):
    rng = random.Random(seed)
    positives = [example for example in examples if example["label"] == 1]
    negatives = [example for example in examples if example["label"] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    pos_val_count = int(len(positives) * val_fraction)
    neg_val_count = int(len(negatives) * val_fraction)

    val_examples = positives[:pos_val_count] + negatives[:neg_val_count]
    train_examples = positives[pos_val_count:] + negatives[neg_val_count:]
    return train_examples, val_examples


def build_vocab(train_examples, max_vocab_size):
    token_counts = Counter()
    for example in train_examples:
        token_counts.update(tokenize_text(str(example["text"])))

    vocab = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX}
    for index, (token, _) in enumerate(token_counts.most_common(max_vocab_size - 2), start=2):
        vocab[token] = index
    return vocab


def encode_text(text, vocab, max_sequence_length):
    token_ids = [vocab.get(token, UNK_INDEX) for token in tokenize_text(text)]
    token_ids = token_ids[:max_sequence_length]
    if len(token_ids) < max_sequence_length:
        token_ids += [PAD_INDEX] * (max_sequence_length - len(token_ids))
    return token_ids


def build_feature_tensor(texts, vocab, max_sequence_length):
    import torch

    encoded_rows = [encode_text(text, vocab, max_sequence_length) for text in texts]
    return torch.tensor(encoded_rows, dtype=torch.long)
