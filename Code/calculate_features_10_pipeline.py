"""
English external-lexicon features.

Features:
- oov_per100
- in_list_per100
- concreteness_coverage_percent
- concreteness_mean_matched
- concreteness_sum_per100

External files required:
1. English frequency list
   - In the original experiment, this was a 5k Project Gutenberg-based
     Wiktionary frequency list:
     https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/English/Project_Gutenberg
   - Expected format: rank word frequency
     Example:
       1   the   56271872
   - The script uses the second column as the known-word list.

2. Brysbaert concreteness ratings
   - In the original experiment, this was:
     Brysbaert et al. concreteness ratings for English words.
   - Expected file: Excel .xlsx
   - Expected columns include Word and Conc.M.

The user should provide:
- an input folder containing cleaned .txt files
- an output CSV path
- a frequency-list path
- a concreteness Excel path
"""

import argparse
import glob
import math
from pathlib import Path

import pandas as pd
import spacy


CHUNK_SIZE = 100
CHAR_BATCH_SIZE = 200_000
SPACY_MODEL = "en_core_web_sm"


FEATURE_NAMES = [
    "oov_per100",
    "in_list_per100",
    "concreteness_coverage_percent",
    "concreteness_mean_matched",
    "concreteness_sum_per100",
]


# ============================================================
# General helpers
# ============================================================

def load_spacy_model(model_name=SPACY_MODEL):
    """Load English spaCy model."""
    try:
        nlp = spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc

    nlp.max_length = 3_000_000
    return nlp


def is_word_token(tok):
    """Alphabetic word tokens only."""
    return tok.is_alpha and not tok.like_num


def lemma(tok):
    return (tok.lemma_ or tok.text).lower()


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


def split_into_chunks(items, chunk_size=CHUNK_SIZE):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# ============================================================
# External lexicon loading
# ============================================================

def load_frequency_list(path):
    """
    Load an English frequency list.

    Expected format:
        rank word frequency

    Example:
        1   the   56271872

    The second column is used as the vocabulary item.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Frequency list not found: {path}")

    vocabulary = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) >= 2:
                vocabulary.add(parts[1].lower())

    return vocabulary


def load_concreteness_lexicon(path):
    """
    Load Brysbaert concreteness ratings from an Excel file.

    Expected columns:
    - Word
    - Conc.M

    The function is permissive about slight column-name variants.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Concreteness file not found: {path}")

    df = pd.read_excel(path)

    word_column = None
    score_column = None

    for column in df.columns:
        normalized = str(column).strip().lower()

        if normalized in {"word", "words"}:
            word_column = column

        if normalized in {"conc.m", "conc_m", "concreteness", "concreteness mean", "mean"}:
            score_column = column

    if word_column is None or score_column is None:
        raise ValueError(
            "Expected columns similar to 'Word' and 'Conc.M'. "
            f"Found columns: {list(df.columns)}"
        )

    def to_float(value):
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(value)

        if isinstance(value, str):
            value = value.strip().replace(",", ".")

            try:
                return float(value)
            except ValueError:
                return math.nan

        return math.nan

    df = df[[word_column, score_column]].dropna()
    df[word_column] = df[word_column].astype(str).str.strip().str.lower()
    df[score_column] = df[score_column].map(to_float)
    df = df.dropna(subset=[word_column, score_column])

    return df.groupby(word_column, as_index=True)[score_column].mean().to_dict()


# ============================================================
# Chunk-level feature computation
# ============================================================

def compute_oov_features(chunk_tokens, frequency_vocabulary):
    """Compute OOV and in-list rates against the supplied frequency vocabulary."""
    n_words = len(chunk_tokens)

    if n_words == 0:
        return {
            "oov_per100": 0.0,
            "in_list_per100": 0.0,
            "oov_total": 0,
            "in_list_total": 0,
        }

    surfaces = [tok.text.lower() for tok in chunk_tokens]

    oov_count = sum(1 for word in surfaces if word not in frequency_vocabulary)
    in_list_count = n_words - oov_count

    scale = 100.0 / n_words

    return {
        "oov_per100": oov_count * scale,
        "in_list_per100": in_list_count * scale,
        "oov_total": oov_count,
        "in_list_total": in_list_count,
    }


def compute_concreteness_features(chunk_tokens, concreteness_lexicon, use_lemma_fallback=True):
    """
    Compute concreteness features for one chunk.

    Matching:
    - first tries lowercased surface form
    - optionally tries lowercased lemma if the surface is not found
    """
    n_words = len(chunk_tokens)

    if n_words == 0:
        return {
            "concreteness_coverage_percent": 0.0,
            "concreteness_mean_matched": 0.0,
            "concreteness_sum_per100": 0.0,
            "concreteness_matched_total": 0,
        }

    matched_scores = []

    for tok in chunk_tokens:
        surface = tok.text.lower()
        score = concreteness_lexicon.get(surface)

        if score is None and use_lemma_fallback:
            score = concreteness_lexicon.get(lemma(tok))

        if score is not None:
            matched_scores.append(score)

    matched_count = len(matched_scores)
    score_sum = float(sum(matched_scores)) if matched_scores else 0.0

    return {
        "concreteness_coverage_percent": (matched_count / n_words) * 100.0,
        "concreteness_mean_matched": score_sum / matched_count if matched_count else 0.0,
        "concreteness_sum_per100": (score_sum / n_words) * 100.0,
        "concreteness_matched_total": matched_count,
    }


# ============================================================
# Book processing
# ============================================================

def process_book(
    text,
    nlp,
    frequency_vocabulary,
    concreteness_lexicon,
    chunk_size=CHUNK_SIZE,
    char_batch_size=CHAR_BATCH_SIZE,
):
    """Process one literary work and return book-level aggregate features."""
    text_batches = split_text_into_char_batches(text, batch_size=char_batch_size)

    word_tokens = []

    for doc in nlp.pipe(text_batches, batch_size=1):
        word_tokens.extend(tok for tok in doc if is_word_token(tok))

    chunks = split_into_chunks(word_tokens, chunk_size=chunk_size)

    collected = {feature: [] for feature in FEATURE_NAMES}

    totals = {
        "oov_total": 0,
        "in_list_total": 0,
        "concreteness_matched_total": 0,
    }

    for chunk in chunks:
        oov = compute_oov_features(chunk, frequency_vocabulary)
        concreteness = compute_concreteness_features(chunk, concreteness_lexicon)

        combined = {**oov, **concreteness}

        for feature in FEATURE_NAMES:
            collected[feature].append(combined[feature])

        totals["oov_total"] += oov["oov_total"]
        totals["in_list_total"] += oov["in_list_total"]
        totals["concreteness_matched_total"] += concreteness["concreteness_matched_total"]

    result = {
        "chunk_size_words": chunk_size,
        "chunk_count_used": len(chunks),
        "token_total": len(word_tokens),
        **totals,
    }

    for feature in FEATURE_NAMES:
        result.update(aggregate_feature(collected[feature], feature))

    return result


# ============================================================
# Corpus processing
# ============================================================

def process_corpus(
    input_dir,
    output_csv,
    frequency_list_path,
    concreteness_path,
    file_pattern="*_cleaned.txt",
):
    """Process all cleaned text files in a folder and save the output CSV."""
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)

    text_files = sorted(glob.glob(str(input_dir / file_pattern)))

    if not text_files:
        raise RuntimeError(f"No files matching '{file_pattern}' were found in: {input_dir}")

    frequency_vocabulary = load_frequency_list(frequency_list_path)
    concreteness_lexicon = load_concreteness_lexicon(concreteness_path)
    nlp = load_spacy_model()

    print(f"Found {len(text_files)} text files.")
    print(f"Loaded frequency vocabulary: {len(frequency_vocabulary)} entries")
    print(f"Loaded concreteness lexicon: {len(concreteness_lexicon)} entries")

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
            frequency_vocabulary=frequency_vocabulary,
            concreteness_lexicon=concreteness_lexicon,
        )

    df = pd.DataFrame(results)

    row_order = [
        "chunk_size_words",
        "chunk_count_used",
        "token_total",

        "oov_total",
        "oov_per100_avg",
        "oov_per100_min",
        "oov_per100_max",
        "oov_per100_sd",

        "in_list_total",
        "in_list_per100_avg",
        "in_list_per100_min",
        "in_list_per100_max",
        "in_list_per100_sd",

        "concreteness_matched_total",
        "concreteness_coverage_percent_avg",
        "concreteness_coverage_percent_min",
        "concreteness_coverage_percent_max",
        "concreteness_coverage_percent_sd",

        "concreteness_mean_matched_avg",
        "concreteness_mean_matched_min",
        "concreteness_mean_matched_max",
        "concreteness_mean_matched_sd",

        "concreteness_sum_per100_avg",
        "concreteness_sum_per100_min",
        "concreteness_sum_per100_max",
        "concreteness_sum_per100_sd",
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
        description="Extract English frequency-list/OOV and concreteness features."
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
        "--frequency_list_path",
        required=True,
        help=(
            "Path to the English frequency list. "
            "Expected format: rank word frequency, with the word in column 2."
        ),
    )

    parser.add_argument(
        "--concreteness_path",
        required=True,
        help=(
            "Path to Brysbaert concreteness ratings Excel file. "
            "Expected columns: Word and Conc.M."
        ),
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
        frequency_list_path=args.frequency_list_path,
        concreteness_path=args.concreteness_path,
        file_pattern=args.file_pattern,
    )


if __name__ == "__main__":
    main()