"""
English readability, lexical-density, and POS-frequency features.

Features:
- words_per_sentence
- syllables_per_sentence
- letters_per_word
- syllables_per_word
- words_different_from_lemma_per100
- punctuation_per100
- verbs_per100
- nouns_per100
- propn_per100
- content_to_function_ratio
- hapax_lemma_per100

The user should provide:
- an input folder containing cleaned .txt files
- an output CSV path
"""

import argparse
import collections
import glob
import math
import re
from pathlib import Path

import pandas as pd
import spacy


CHUNK_SIZE = 100
CHAR_BATCH_SIZE = 200_000
SPACY_MODEL = "en_core_web_sm"

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

VOWELS_EN = "aeiouy"
VOWEL_RE_EN = re.compile(rf"[{VOWELS_EN}]+", re.IGNORECASE)


FEATURE_NAMES = [
    "words_per_sentence",
    "syllables_per_sentence",
    "letters_per_word",
    "syllables_per_word",
    "words_different_from_lemma_per100",
    "punctuation_per100",
    "verbs_per100",
    "nouns_per100",
    "propn_per100",
    "content_to_function_ratio",
    "hapax_lemma_per100",
]


# ============================================================
# Helpers
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
    """Alphabetic word tokens; excludes punctuation, spaces, symbols, and numbers."""
    if tok.is_space or tok.is_punct or tok.like_num or tok.pos_ == "SYM":
        return False
    return any(ch.isalpha() for ch in tok.text)


def letters_only(word):
    """Keep only alphabetic characters."""
    return "".join(ch for ch in word if ch.isalpha())


def syllables_en(word):
    """
    Simple English syllable heuristic:
    - count vowel groups
    - subtract final silent -e, except -le after a consonant
    - return at least 1 for alphabetic words
    """
    word = letters_only(word.lower())

    if not word:
        return 0

    groups = VOWEL_RE_EN.findall(word)
    syllable_count = len(groups)

    if syllable_count > 1:
        if word.endswith("e") and not re.search(r"[aeiouy]le$", word):
            syllable_count -= 1

    return max(1, syllable_count)


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


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 9999.0 if numerator > 0 else 0.0
    return numerator / denominator


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
# Chunk-level features
# ============================================================

def compute_readability_features(chunk_tokens):
    """Compute sentence/word length features for one 100-word chunk."""
    if not chunk_tokens:
        return {
            "words_per_sentence": 0.0,
            "syllables_per_sentence": 0.0,
            "letters_per_word": 0.0,
            "syllables_per_word": 0.0,
            "n_sentences": 0,
        }

    sentence_to_words = {}

    for tok in chunk_tokens:
        sentence_to_words.setdefault(tok.sent.start, []).append(tok)

    sentence_word_counts = []
    sentence_syllable_counts = []
    letters_per_word_values = []
    syllables_per_word_values = []

    for sentence_words in sentence_to_words.values():
        sentence_word_counts.append(len(sentence_words))
        sentence_syllables = 0

        for tok in sentence_words:
            letters = len(letters_only(tok.text))
            syllables = syllables_en(tok.text)

            letters_per_word_values.append(letters)
            syllables_per_word_values.append(syllables)
            sentence_syllables += syllables

        sentence_syllable_counts.append(sentence_syllables)

    n_sentences = len(sentence_word_counts)
    n_words = len(chunk_tokens)

    return {
        "words_per_sentence": safe_avg(sentence_word_counts),
        "syllables_per_sentence": safe_avg(sentence_syllable_counts),
        "letters_per_word": sum(letters_per_word_values) / n_words if n_words else 0.0,
        "syllables_per_word": sum(syllables_per_word_values) / n_words if n_words else 0.0,
        "n_sentences": n_sentences,
    }


def compute_pos_frequency_features(chunk_tokens):
    """Compute lemma-difference and POS-frequency features for one chunk."""
    n_words = len(chunk_tokens)

    if n_words == 0:
        return {
            "words_different_from_lemma_per100": 0.0,
            "verbs_per100": 0.0,
            "nouns_per100": 0.0,
            "propn_per100": 0.0,
            "words_different_from_lemma_total": 0,
            "verbs_total": 0,
            "nouns_total": 0,
            "propn_total": 0,
        }

    n_diff_lemma = sum(
        1 for tok in chunk_tokens
        if lemma(tok) != tok.text.lower()
    )

    n_verbs = sum(1 for tok in chunk_tokens if tok.pos_ == "VERB")
    n_nouns = sum(1 for tok in chunk_tokens if tok.pos_ == "NOUN")
    n_propn = sum(1 for tok in chunk_tokens if tok.pos_ == "PROPN")

    scale = 100.0 / n_words

    return {
        "words_different_from_lemma_per100": n_diff_lemma * scale,
        "verbs_per100": n_verbs * scale,
        "nouns_per100": n_nouns * scale,
        "propn_per100": n_propn * scale,
        "words_different_from_lemma_total": n_diff_lemma,
        "verbs_total": n_verbs,
        "nouns_total": n_nouns,
        "propn_total": n_propn,
    }


def compute_punctuation_per100(chunk_tokens):
    """
    Approximate punctuation frequency for a word chunk.

    Punctuation is counted between the first and last word token of the chunk.
    """
    if not chunk_tokens:
        return 0.0, 0

    first_token = chunk_tokens[0]
    last_token = chunk_tokens[-1]
    doc = first_token.doc

    start_char = first_token.idx
    end_char = last_token.idx + len(last_token.text)

    punctuation_count = sum(
        1 for tok in doc
        if tok.is_punct and start_char <= tok.idx < end_char
    )

    return (punctuation_count / len(chunk_tokens)) * 100.0, punctuation_count


def compute_content_function_and_hapax(chunk_tokens):
    """Compute content/function ratio and chunk-level hapax lemma percentage."""
    n_words = len(chunk_tokens)

    if n_words == 0:
        return {
            "content_to_function_ratio": 0.0,
            "hapax_lemma_per100": 0.0,
            "content_total": 0,
            "function_total": 0,
            "hapax_lemma_total": 0,
        }

    n_content = sum(1 for tok in chunk_tokens if tok.pos_ in CONTENT_POS)
    n_function = n_words - n_content

    lemmas = [lemma(tok) for tok in chunk_tokens]
    lemma_counts = collections.Counter(lemmas)
    n_hapax = sum(1 for count in lemma_counts.values() if count == 1)

    return {
        "content_to_function_ratio": safe_ratio(n_content, n_function),
        "hapax_lemma_per100": (n_hapax / n_words) * 100.0,
        "content_total": n_content,
        "function_total": n_function,
        "hapax_lemma_total": n_hapax,
    }


# ============================================================
# Book processing
# ============================================================

def process_book(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """Process one literary work and return book-level aggregate features."""
    text_batches = split_text_into_char_batches(text, batch_size=char_batch_size)

    word_tokens = []

    for doc in nlp.pipe(text_batches, batch_size=1):
        word_tokens.extend(tok for tok in doc if is_word_token(tok))

    chunks = split_into_chunks(word_tokens, chunk_size=chunk_size)

    collected = {name: [] for name in FEATURE_NAMES}

    totals = {
        "sentence_total_used": 0,
        "words_different_from_lemma_total": 0,
        "punctuation_total": 0,
        "verbs_total": 0,
        "nouns_total": 0,
        "propn_total": 0,
        "content_total": 0,
        "function_total": 0,
        "hapax_lemma_total": 0,
    }

    for chunk in chunks:
        readability = compute_readability_features(chunk)
        posfreq = compute_pos_frequency_features(chunk)
        punctuation_per100, punctuation_total = compute_punctuation_per100(chunk)
        lexical_density = compute_content_function_and_hapax(chunk)

        combined = {
            **readability,
            **posfreq,
            "punctuation_per100": punctuation_per100,
            **lexical_density,
        }

        for feature in FEATURE_NAMES:
            collected[feature].append(combined[feature])

        totals["sentence_total_used"] += readability["n_sentences"]
        totals["words_different_from_lemma_total"] += posfreq["words_different_from_lemma_total"]
        totals["punctuation_total"] += punctuation_total
        totals["verbs_total"] += posfreq["verbs_total"]
        totals["nouns_total"] += posfreq["nouns_total"]
        totals["propn_total"] += posfreq["propn_total"]
        totals["content_total"] += lexical_density["content_total"]
        totals["function_total"] += lexical_density["function_total"]
        totals["hapax_lemma_total"] += lexical_density["hapax_lemma_total"]

    result = {
        "chunk_size_words": chunk_size,
        "chunk_count_used": len(chunks),
        "token_total": len(word_tokens),
        **totals,
    }

    for feature in FEATURE_NAMES:
        result.update(aggregate_feature(collected[feature], feature))

    return result


def process_corpus(input_dir, output_csv, file_pattern="*_cleaned.txt"):
    """Process all cleaned text files in a folder and save the output CSV."""
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)

    text_files = sorted(glob.glob(str(input_dir / file_pattern)))

    if not text_files:
        raise RuntimeError(f"No files matching '{file_pattern}' were found in: {input_dir}")

    nlp = load_spacy_model()
    results = {}

    print(f"Found {len(text_files)} text files.")

    for idx, filepath in enumerate(text_files, start=1):
        filepath = Path(filepath)
        work_name = filepath.name.replace("_cleaned.txt", "")

        print(f"[{idx}/{len(text_files)}] Processing: {work_name}")

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        results[work_name] = process_book(text, nlp=nlp)

    df = pd.DataFrame(results)

    row_order = [
        "chunk_size_words",
        "chunk_count_used",
        "token_total",
        "sentence_total_used",

        "words_per_sentence_avg",
        "words_per_sentence_min",
        "words_per_sentence_max",
        "words_per_sentence_sd",

        "syllables_per_sentence_avg",
        "syllables_per_sentence_min",
        "syllables_per_sentence_max",
        "syllables_per_sentence_sd",

        "letters_per_word_avg",
        "letters_per_word_min",
        "letters_per_word_max",
        "letters_per_word_sd",

        "syllables_per_word_avg",
        "syllables_per_word_min",
        "syllables_per_word_max",
        "syllables_per_word_sd",

        "words_different_from_lemma_total",
        "words_different_from_lemma_per100_avg",
        "words_different_from_lemma_per100_min",
        "words_different_from_lemma_per100_max",
        "words_different_from_lemma_per100_sd",

        "punctuation_total",
        "punctuation_per100_avg",
        "punctuation_per100_min",
        "punctuation_per100_max",
        "punctuation_per100_sd",

        "verbs_total",
        "verbs_per100_avg",
        "verbs_per100_min",
        "verbs_per100_max",
        "verbs_per100_sd",

        "nouns_total",
        "nouns_per100_avg",
        "nouns_per100_min",
        "nouns_per100_max",
        "nouns_per100_sd",

        "propn_total",
        "propn_per100_avg",
        "propn_per100_min",
        "propn_per100_max",
        "propn_per100_sd",

        "content_total",
        "function_total",
        "content_to_function_ratio_avg",
        "content_to_function_ratio_min",
        "content_to_function_ratio_max",
        "content_to_function_ratio_sd",

        "hapax_lemma_total",
        "hapax_lemma_per100_avg",
        "hapax_lemma_per100_min",
        "hapax_lemma_per100_max",
        "hapax_lemma_per100_sd",
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
        description="Extract English readability, lexical-density, and POS-frequency features."
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
        "--file_pattern",
        default="*_cleaned.txt",
        help="Pattern used to select input files. Default: *_cleaned.txt",
    )

    args = parser.parse_args()

    process_corpus(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        file_pattern=args.file_pattern,
    )


if __name__ == "__main__":
    main()