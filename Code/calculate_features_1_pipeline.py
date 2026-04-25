"""
English tense/aspect, adverbial, and pronoun features.

Features calculated per 100-word chunk, then aggregated at book level:
- past_per100
- present_per100
- perfect_per100
- place_adverbials_per100
- time_adverbials_per100
- pronouns_total_per100
- pronouns_1p_per100
- pronouns_2p_per100
- pronouns_3p_per100
- pronouns_demonstrative_per100
- pronouns_indefinite_per100

For each feature, the script outputs:
- average
- minimum
- maximum
- standard deviation

The user should provide:
- an input folder containing cleaned .txt literary works
- an output CSV path

Example use:
python english_grammar_features.py \
    --input_dir /path/to/cleaned_texts \
    --output_csv /path/to/output/english_grammar_features.csv
"""

import argparse
import glob
import math
import os
from pathlib import Path

import pandas as pd
import spacy


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 100
CHAR_BATCH_SIZE = 200_000
SPACY_MODEL = "en_core_web_sm"


# ============================================================
# English lexical inventories
# ============================================================

PLACE_ADV_LEMMAS = {
    "here", "there", "outside", "inside", "above", "below", "nearby", "far",
    "everywhere", "nowhere", "indoors", "outdoors", "around", "behind",
    "ahead", "upstairs", "downstairs", "abroad", "near"
}

TIME_ADV_LEMMAS = {
    "yesterday", "today", "tomorrow", "soon", "early", "late", "long", "often",
    "sometimes", "rarely", "always", "already", "still", "then", "afterwards",
    "before", "later", "frequently", "eventually", "next"
}

EN_FIRST = {"i", "me", "we", "us", "myself", "ourselves"}
EN_SECOND = {"you", "yourself", "yourselves"}
EN_THIRD = {
    "he", "him", "she", "her", "it", "they", "them",
    "himself", "herself", "itself", "themselves"
}
EN_DEMO = {"this", "that", "these", "those"}
EN_INDEF = {
    "someone", "somebody", "something", "anyone", "anybody", "anything",
    "no one", "nobody", "nothing", "everyone", "everybody", "everything",
    "each", "none", "another", "others", "few", "many", "several",
    "both", "either", "neither", "one", "all", "some", "any", "most"
}

EN_PRONOUN_LEXICON = EN_FIRST | EN_SECOND | EN_THIRD | EN_DEMO | EN_INDEF

SKIP_POS = {"PART", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PUNCT"}
SKIP_LEMMAS = {"not", "n't", "never", "no"}


# ============================================================
# General helpers
# ============================================================

def load_spacy_model(model_name=SPACY_MODEL):
    """Load the English spaCy model used for POS, lemma, and morphology."""
    try:
        nlp = spacy.load(model_name, disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc

    nlp.max_length = 3_000_000
    return nlp


def is_word_token(tok):
    """Keep words and exclude spaces/punctuation."""
    return not tok.is_space and not tok.is_punct


def morph_has(tok, key, value):
    """Check whether a spaCy token has a specific morphological value."""
    values = tok.morph.get(key)
    return isinstance(values, list) and value in values


def morph_any(tok, key, candidates):
    """Check whether a spaCy token has any value from a candidate set."""
    values = tok.morph.get(key)
    return isinstance(values, list) and any(v in candidates for v in values)


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
    """Return avg/min/max/sd for one chunk-level feature."""
    return {
        f"{prefix}_avg": round(safe_avg(values), 4),
        f"{prefix}_min": round(safe_min(values), 4),
        f"{prefix}_max": round(safe_max(values), 4),
        f"{prefix}_sd": round(safe_sd(values), 4),
    }


# ============================================================
# Long-text processing
# ============================================================

def split_text_into_char_batches(text, batch_size=CHAR_BATCH_SIZE):
    """Split long texts into manageable batches, preferably at whitespace."""
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


def extract_word_tokens_from_text(text, nlp, batch_size=CHAR_BATCH_SIZE):
    """Apply spaCy to a long text and return word tokens only."""
    text_batches = split_text_into_char_batches(text, batch_size=batch_size)
    word_tokens = []

    for doc in nlp.pipe(text_batches, batch_size=1):
        word_tokens.extend(tok for tok in doc if is_word_token(tok))

    return word_tokens


def split_into_word_chunks(word_tokens, chunk_size=CHUNK_SIZE, include_last_partial_chunk=True):
    """Split a token list into fixed-size word chunks."""
    chunks = []

    for i in range(0, len(word_tokens), chunk_size):
        chunk = word_tokens[i:i + chunk_size]

        if len(chunk) < chunk_size and not include_last_partial_chunk:
            continue

        chunks.append(chunk)

    return chunks


# ============================================================
# Feature extraction: tense and aspect
# ============================================================

def count_past_tense(chunk_tokens):
    """Count verbs and auxiliaries morphologically marked as past tense."""
    return sum(
        1 for tok in chunk_tokens
        if tok.pos_ in {"VERB", "AUX"} and morph_any(tok, "Tense", {"Past"})
    )


def count_present_tense(chunk_tokens):
    """Count verbs and auxiliaries morphologically marked as present tense."""
    return sum(
        1 for tok in chunk_tokens
        if tok.pos_ in {"VERB", "AUX"} and morph_has(tok, "Tense", "Pres")
    )


def count_perfect_aspect(chunk_tokens, window=3):
    """
    Approximate English perfect aspect.

    Counts finite forms of auxiliary 'have' followed shortly by a participle,
    allowing a small number of intervening function words or negation.
    """
    count = 0

    for i, tok in enumerate(chunk_tokens):
        if tok.pos_ != "AUX":
            continue
        if tok.lemma_.lower() != "have":
            continue
        if not morph_has(tok, "VerbForm", "Fin"):
            continue

        j = i + 1
        steps = 0

        while j < len(chunk_tokens) and steps < window:
            candidate = chunk_tokens[j]

            if candidate.is_space:
                j += 1
                continue

            if candidate.pos_ in SKIP_POS or candidate.lemma_.lower() in SKIP_LEMMAS:
                j += 1
                steps += 1
                continue

            if morph_has(candidate, "VerbForm", "Part"):
                count += 1

            break

    return count


# ============================================================
# Feature extraction: place/time adverbials
# ============================================================

def count_place_adverbials(chunk_tokens):
    """Count selected place adverbs using lemma matching and POS=ADV."""
    return sum(
        1 for tok in chunk_tokens
        if tok.pos_ == "ADV" and tok.lemma_.lower() in PLACE_ADV_LEMMAS
    )


def count_time_adverbials(chunk_tokens):
    """Count selected time adverbs using lemma matching and POS=ADV."""
    return sum(
        1 for tok in chunk_tokens
        if tok.pos_ == "ADV" and tok.lemma_.lower() in TIME_ADV_LEMMAS
    )


# ============================================================
# Feature extraction: pronouns
# ============================================================

def count_pronoun_categories(chunk_tokens):
    """
    Count English pronoun categories.

    Categories are assigned exclusively in this order:
    first person, second person, third person, demonstrative, indefinite.
    """
    counts = {
        "pronouns_total_per100": 0,
        "pronouns_1p_per100": 0,
        "pronouns_2p_per100": 0,
        "pronouns_3p_per100": 0,
        "pronouns_demonstrative_per100": 0,
        "pronouns_indefinite_per100": 0,
    }

    for tok in chunk_tokens:
        lemma = tok.lemma_.lower()
        text = tok.text.lower()

        if tok.pos_ != "PRON" and lemma not in EN_PRONOUN_LEXICON and text not in EN_PRONOUN_LEXICON:
            continue

        key = lemma if lemma in EN_PRONOUN_LEXICON else text

        if key in EN_FIRST:
            counts["pronouns_1p_per100"] += 1
        elif key in EN_SECOND:
            counts["pronouns_2p_per100"] += 1
        elif key in EN_THIRD:
            counts["pronouns_3p_per100"] += 1
        elif key in EN_DEMO:
            counts["pronouns_demonstrative_per100"] += 1
        elif key in EN_INDEF:
            counts["pronouns_indefinite_per100"] += 1
        else:
            continue

        counts["pronouns_total_per100"] += 1

    return counts


# ============================================================
# Chunk-level feature calculation
# ============================================================

FEATURE_NAMES = [
    "past_per100",
    "present_per100",
    "perfect_per100",
    "place_adverbials_per100",
    "time_adverbials_per100",
    "pronouns_total_per100",
    "pronouns_1p_per100",
    "pronouns_2p_per100",
    "pronouns_3p_per100",
    "pronouns_demonstrative_per100",
    "pronouns_indefinite_per100",
]


def calculate_chunk_features(chunk_tokens):
    """Calculate all selected features for one word chunk."""
    word_count = len(chunk_tokens)

    if word_count == 0:
        return None

    raw_counts = {
        "past_per100": count_past_tense(chunk_tokens),
        "present_per100": count_present_tense(chunk_tokens),
        "perfect_per100": count_perfect_aspect(chunk_tokens),
        "place_adverbials_per100": count_place_adverbials(chunk_tokens),
        "time_adverbials_per100": count_time_adverbials(chunk_tokens),
    }

    raw_counts.update(count_pronoun_categories(chunk_tokens))

    return {
        feature: (count / word_count) * 100.0
        for feature, count in raw_counts.items()
    }


# ============================================================
# Book-level processing
# ============================================================

def process_book(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """Process one literary work and return book-level aggregate features."""
    word_tokens = extract_word_tokens_from_text(
        text,
        nlp=nlp,
        batch_size=char_batch_size,
    )

    chunks = split_into_word_chunks(
        word_tokens,
        chunk_size=chunk_size,
        include_last_partial_chunk=True,
    )

    collected = {feature: [] for feature in FEATURE_NAMES}

    for chunk in chunks:
        chunk_values = calculate_chunk_features(chunk)

        if chunk_values is None:
            continue

        for feature in FEATURE_NAMES:
            collected[feature].append(chunk_values[feature])

    result = {
        "chunk_size_words": chunk_size,
        "chunk_count_used": len(chunks),
        "token_total": len(word_tokens),
    }

    for feature in FEATURE_NAMES:
        result.update(aggregate_feature(collected[feature], feature))

    return result


def process_corpus(input_dir, output_csv, file_pattern="*_cleaned.txt"):
    """
    Process all cleaned text files in a folder and save a CSV table.

    The input directory should contain one cleaned .txt file per literary work.
    By default, files are expected to end in '_cleaned.txt'.
    """
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)

    text_files = sorted(glob.glob(str(input_dir / file_pattern)))

    if not text_files:
        raise RuntimeError(
            f"No files matching '{file_pattern}' were found in: {input_dir}"
        )

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
    ]

    for feature in FEATURE_NAMES:
        row_order.extend([
            f"{feature}_avg",
            f"{feature}_min",
            f"{feature}_max",
            f"{feature}_sd",
        ])

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
        description="Extract English grammatical features from literary texts."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing cleaned .txt files. Example: /path/to/cleaned_texts",
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