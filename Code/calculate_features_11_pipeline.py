"""
English neural-model-based features.

Features:
- humor_prob
- nonhumor_prob
- humor_label
- metaphor_word_percent
- metaphor_prob_mean
- metaphor_words_total

Models used in the original experiment:
- Humor: mohameddhiab/humor-no-humor
- Metaphor: lwachowiak/Metaphor-Detection-XLMR

The user should provide:
- an input folder containing cleaned .txt files
- an output CSV path

This version processes the whole corpus by default.
"""

import argparse
import glob
import math
from pathlib import Path

import pandas as pd
import spacy
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
)


CHUNK_SIZE = 100
CHAR_BATCH_SIZE = 200_000
SPACY_MODEL = "en_core_web_sm"

DEFAULT_HUMOR_MODEL = "mohameddhiab/humor-no-humor"
DEFAULT_METAPHOR_MODEL = "lwachowiak/Metaphor-Detection-XLMR"


# ============================================================
# Helpers
# ============================================================

def load_spacy_model(model_name=SPACY_MODEL):
    """Load English spaCy model for tokenisation."""
    try:
        nlp = spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc

    nlp.max_length = 4_000_000
    return nlp


def is_word_token(tok):
    """Alphabetic word tokens only."""
    return tok.is_alpha and not tok.like_num


def safe_avg(values):
    return sum(values) / len(values) if values else 0.0


def safe_min(values):
    return min(values) if values else 0.0


def safe_max(values):
    return max(values) if values else 0.0


def safe_sd(values):
    if len(values) <= 1:
        return 0.0
    mean = safe_avg(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def aggregate_feature(values, prefix):
    return {
        f"{prefix}_avg": round(safe_avg(values), 4),
        f"{prefix}_min": round(safe_min(values), 4),
        f"{prefix}_max": round(safe_max(values), 4),
        f"{prefix}_sd": round(safe_sd(values), 4),
    }


def split_text_into_char_batches(text, batch_size=CHAR_BATCH_SIZE):
    """Split long texts into smaller batches, preferably at whitespace."""
    batches = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + batch_size, n)

        if end < n:
            split_pos = text.rfind(" ", start, end)
            if split_pos == -1 or split_pos <= start:
                split_pos = end
        else:
            split_pos = end

        batches.append(text[start:split_pos])
        start = split_pos

    return batches


def text_to_word_chunks(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """Convert a text into 100-word chunks."""
    words = []

    for doc in nlp.pipe(split_text_into_char_batches(text, char_batch_size), batch_size=1):
        words.extend(tok.text for tok in doc if is_word_token(tok))

    chunks = [
        words[i:i + chunk_size]
        for i in range(0, len(words), chunk_size)
        if words[i:i + chunk_size]
    ]

    return chunks, len(words)


# ============================================================
# Humor model
# ============================================================

def load_humor_classifier(model_name=DEFAULT_HUMOR_MODEL):
    """Load Hugging Face humor/no-humor classifier."""
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        top_k=None,
        device=device,
    )

    print(f"Loaded humor model: {model_name}")
    print(f"Humor model device: {'GPU' if device == 0 else 'CPU'}")

    return classifier


def extract_humor_scores(raw_prediction):
    """
    Extract humor and non-humor probabilities from common HF output formats.
    """
    humor_prob = 0.0
    nonhumor_prob = 0.0

    if (
        isinstance(raw_prediction, list)
        and raw_prediction
        and isinstance(raw_prediction[0], list)
    ):
        raw_prediction = raw_prediction[0]

    if (
        isinstance(raw_prediction, list)
        and raw_prediction
        and isinstance(raw_prediction[0], dict)
    ):
        score_map = {}

        for item in raw_prediction:
            label = str(item.get("label", "")).strip().upper()
            score = float(item.get("score", 0.0))
            score_map[label] = score

        humor_prob = score_map.get("HUMOR", score_map.get("LABEL_1", 0.0))
        nonhumor_prob = score_map.get("NO_HUMOR", score_map.get("LABEL_0", 0.0))

        if humor_prob == 0.0 and nonhumor_prob > 0.0:
            humor_prob = 1.0 - nonhumor_prob

        if nonhumor_prob == 0.0 and humor_prob > 0.0:
            nonhumor_prob = 1.0 - humor_prob

        return humor_prob, nonhumor_prob

    if isinstance(raw_prediction, dict):
        label = str(raw_prediction.get("label", "")).strip().upper()
        score = float(raw_prediction.get("score", 0.0))

        if label in {"HUMOR", "LABEL_1"}:
            return score, 1.0 - score

        return 1.0 - score, score

    return humor_prob, nonhumor_prob


def compute_humor_features(chunk_words, humor_classifier):
    """Compute humor probabilities for one chunk."""
    chunk_text = " ".join(chunk_words)
    raw_prediction = humor_classifier(chunk_text)

    humor_prob, nonhumor_prob = extract_humor_scores(raw_prediction)
    humor_label = 1 if humor_prob >= 0.5 else 0

    return {
        "humor_prob": humor_prob,
        "nonhumor_prob": nonhumor_prob,
        "humor_label": humor_label,
    }


# ============================================================
# Metaphor model
# ============================================================

def load_metaphor_model(model_id=DEFAULT_METAPHOR_MODEL):
    """Load token-classification metaphor model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.eval().to(device)

    id2label = model.config.id2label

    metaphor_label_id = None

    for label_id, label_name in id2label.items():
        if str(label_name).upper() == "LABEL_1":
            metaphor_label_id = int(label_id)
            break

    if metaphor_label_id is None:
        raise RuntimeError(f"Could not find LABEL_1 in model labels: {id2label}")

    print(f"Loaded metaphor model: {model_id}")
    print(f"Metaphor model labels: {id2label}")
    print(f"Metaphor model device: {device}")

    return tokenizer, model, metaphor_label_id, device


def compute_metaphor_features(
    chunk_words,
    tokenizer,
    model,
    metaphor_label_id,
    device,
    threshold=0.5,
):
    """Compute metaphor probability and positive word percentage for one chunk."""
    if not chunk_words:
        return {
            "metaphor_word_percent": 0.0,
            "metaphor_prob_mean": 0.0,
            "metaphor_words_total": 0,
        }

    encoded = tokenizer(
        chunk_words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
    )

    word_ids = encoded.word_ids(batch_index=0)

    with torch.no_grad():
        output = model(**{key: value.to(device) for key, value in encoded.items()})
        probabilities = torch.softmax(output.logits, dim=-1)[0].detach().cpu()

    word_to_probs = {}

    for token_index, word_index in enumerate(word_ids):
        if word_index is None:
            continue

        metaphor_prob = float(probabilities[token_index, metaphor_label_id])
        word_to_probs.setdefault(word_index, []).append(metaphor_prob)

    word_level_probs = []

    for word_index in range(len(chunk_words)):
        subword_probs = word_to_probs.get(word_index, [])
        prob = safe_avg(subword_probs) if subword_probs else 0.0
        word_level_probs.append(prob)

    word_level_labels = [
        1 if prob >= threshold else 0
        for prob in word_level_probs
    ]

    metaphor_words_total = sum(word_level_labels)

    return {
        "metaphor_word_percent": (metaphor_words_total / len(chunk_words)) * 100.0,
        "metaphor_prob_mean": safe_avg(word_level_probs),
        "metaphor_words_total": metaphor_words_total,
    }


# ============================================================
# Book processing
# ============================================================

def process_book(
    text,
    nlp,
    humor_classifier,
    metaphor_tokenizer,
    metaphor_model,
    metaphor_label_id,
    metaphor_device,
    chunk_size=CHUNK_SIZE,
    metaphor_threshold=0.5,
):
    """Process one literary work and return book-level aggregate neural features."""
    chunks, token_total = text_to_word_chunks(text, nlp=nlp, chunk_size=chunk_size)

    collected = {
        "humor_prob": [],
        "nonhumor_prob": [],
        "humor_label": [],
        "metaphor_word_percent": [],
        "metaphor_prob_mean": [],
        "metaphor_words_total": [],
    }

    humor_chunks_positive_total = 0
    metaphor_words_positive_total = 0

    for chunk_words in chunks:
        humor = compute_humor_features(chunk_words, humor_classifier)
        metaphor = compute_metaphor_features(
            chunk_words=chunk_words,
            tokenizer=metaphor_tokenizer,
            model=metaphor_model,
            metaphor_label_id=metaphor_label_id,
            device=metaphor_device,
            threshold=metaphor_threshold,
        )

        for key, value in humor.items():
            collected[key].append(value)

        for key, value in metaphor.items():
            collected[key].append(value)

        humor_chunks_positive_total += humor["humor_label"]
        metaphor_words_positive_total += metaphor["metaphor_words_total"]

    result = {
        "chunk_size_words": chunk_size,
        "chunk_count_used": len(chunks),
        "token_total": token_total,
        "humor_chunks_positive_total": humor_chunks_positive_total,
        "metaphor_words_positive_total": metaphor_words_positive_total,
    }

    for feature_name, values in collected.items():
        result.update(aggregate_feature(values, feature_name))

    return result


# ============================================================
# Corpus processing
# ============================================================

def process_corpus(
    input_dir,
    output_csv,
    humor_model_name=DEFAULT_HUMOR_MODEL,
    metaphor_model_id=DEFAULT_METAPHOR_MODEL,
    metaphor_threshold=0.5,
    file_pattern="*_cleaned.txt",
):
    """Process all cleaned text files in the corpus."""
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)

    text_files = sorted(glob.glob(str(input_dir / file_pattern)))

    if not text_files:
        raise RuntimeError(f"No files matching '{file_pattern}' were found in: {input_dir}")

    nlp = load_spacy_model()
    humor_classifier = load_humor_classifier(humor_model_name)

    (
        metaphor_tokenizer,
        metaphor_model,
        metaphor_label_id,
        metaphor_device,
    ) = load_metaphor_model(metaphor_model_id)

    print(f"\nFound {len(text_files)} cleaned text files.")
    print("Processing the full corpus.\n")

    results = {}

    for idx, filepath in enumerate(text_files, start=1):
        filepath = Path(filepath)
        work_name = filepath.name.replace("_cleaned.txt", "")

        print(f"[{idx}/{len(text_files)}] Processing: {work_name}")

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        results[work_name] = process_book(
            text=text,
            nlp=nlp,
            humor_classifier=humor_classifier,
            metaphor_tokenizer=metaphor_tokenizer,
            metaphor_model=metaphor_model,
            metaphor_label_id=metaphor_label_id,
            metaphor_device=metaphor_device,
            chunk_size=CHUNK_SIZE,
            metaphor_threshold=metaphor_threshold,
        )

    df = pd.DataFrame(results)

    row_order = [
        "chunk_size_words",
        "chunk_count_used",
        "token_total",

        "humor_chunks_positive_total",
        "humor_prob_avg",
        "humor_prob_min",
        "humor_prob_max",
        "humor_prob_sd",

        "nonhumor_prob_avg",
        "nonhumor_prob_min",
        "nonhumor_prob_max",
        "nonhumor_prob_sd",

        "humor_label_avg",
        "humor_label_min",
        "humor_label_max",
        "humor_label_sd",

        "metaphor_words_positive_total",
        "metaphor_word_percent_avg",
        "metaphor_word_percent_min",
        "metaphor_word_percent_max",
        "metaphor_word_percent_sd",

        "metaphor_prob_mean_avg",
        "metaphor_prob_mean_min",
        "metaphor_prob_mean_max",
        "metaphor_prob_mean_sd",

        "metaphor_words_total_avg",
        "metaphor_words_total_min",
        "metaphor_words_total_max",
        "metaphor_words_total_sd",
    ]

    df = df.loc[row_order]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, encoding="utf-8")

    print("\nDone.")
    print(f"Saved CSV to: {output_csv}")

    return df


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract English neural humor and metaphor features."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing cleaned .txt files.",
    )

    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path where the output CSV should be saved.",
    )

    parser.add_argument(
        "--humor_model_name",
        default=DEFAULT_HUMOR_MODEL,
        help=f"Humor classifier model name. Default: {DEFAULT_HUMOR_MODEL}",
    )

    parser.add_argument(
        "--metaphor_model_id",
        default=DEFAULT_METAPHOR_MODEL,
        help=f"Metaphor token-classification model ID. Default: {DEFAULT_METAPHOR_MODEL}",
    )

    parser.add_argument(
        "--metaphor_threshold",
        type=float,
        default=0.5,
        help="Threshold for converting metaphor probabilities to binary labels.",
    )

    parser.add_argument(
        "--file_pattern",
        default="*_cleaned.txt",
        help="Pattern used to select input files. Default: *_cleaned.txt",
    )

    args = parser.parse_args()

    process_corpus(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        humor_model_name=args.humor_model_name,
        metaphor_model_id=args.metaphor_model_id,
        metaphor_threshold=args.metaphor_threshold,
        file_pattern=args.file_pattern,
    )


if __name__ == "__main__":
    main()