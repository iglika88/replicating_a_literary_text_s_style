"""
English question, nominalisation, and passive features.

Features calculated per 100-word chunk, then aggregated at book level:

Direct WH-question features:
- direct_wh_questions_per100
- direct_wh_questions_total

Nominalisation features:
- nominal_suffix_per100
- gerund_nominals_ing_per100

Passive construction features:
- by_passives_per100
- agentless_passives_per100
- by_passives_total
- agentless_passives_total

For each per-100 feature, the script outputs:
- average
- minimum
- maximum
- standard deviation

The user should provide:
- an input folder containing cleaned .txt literary works
- an output CSV path

Example use:
python english_sentence_nominal_passive_features.py \
    --input_dir /path/to/cleaned_texts \
    --output_csv /path/to/output/english_sentence_nominal_passive_features.csv
"""

import argparse
import glob
import math
import os
import re
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
# English inventories
# ============================================================

EN_WH_SINGLE = {
    "who", "whom", "whose", "what", "which", "where", "when", "why", "how"
}

EN_WH_MULTI_RE = [
    re.compile(r"\bwhat\s+kind\s+of\b", re.I),
    re.compile(r"\bhow\s+(many|much|long|often|far)\b", re.I),
]

EN_NOMINAL_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance",
    "ence", "ship", "hood", "al", "ure", "ery"
)

PASSIVE_SKIP_POS = {"PART", "ADV", "DET", "PRON", "PUNCT", "CCONJ", "SCONJ"}
NEGATION_LEMMAS = {"not", "n't", "never"}
AUX_PASSIVE_LEMMAS = {"be", "get"}


# ============================================================
# General helpers
# ============================================================

def load_spacy_model(model_name=SPACY_MODEL):
    """
    Load the English spaCy model.

    The parser and NER are disabled for speed. Sentence segmentation is provided
    by spaCy's rule-based sentencizer.
    """
    try:
        nlp = spacy.load(model_name, disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    nlp.max_length = 3_000_000
    return nlp


def is_word_token(tok):
    """Keep words and exclude spaces/punctuation."""
    return not tok.is_space and not tok.is_punct


def morph_has(tok, key, value):
    """Check whether a spaCy token has a specific morphological value."""
    values = tok.morph.get(key)
    return isinstance(values, list) and value in values


def is_participle(tok):
    """Robust participle check for English."""
    return morph_has(tok, "VerbForm", "Part") or getattr(tok, "tag_", "") == "VBN"


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


def split_into_word_chunks(word_tokens, chunk_size=CHUNK_SIZE):
    """Split a token list into fixed-size word chunks."""
    return [
        word_tokens[i:i + chunk_size]
        for i in range(0, len(word_tokens), chunk_size)
    ]


def get_last_chunk_size(token_total, chunk_index, chunk_count, chunk_size):
    """Return the actual word count of a chunk, accounting for the final partial chunk."""
    if chunk_index < chunk_count - 1:
        return chunk_size

    current_chunk_size = token_total - (chunk_index * chunk_size)
    return current_chunk_size if current_chunk_size > 0 else chunk_size


# ============================================================
# Direct WH-question detection
# ============================================================

def sentence_ends_with_question_mark(sentence_text):
    """Check whether a sentence ends with a question mark."""
    return sentence_text.rstrip().endswith("?")


def contains_wh_expression(sentence):
    """Check whether a sentence contains a WH word or selected multi-word WH expression."""
    sentence_text = sentence.text.strip()
    sentence_lower = sentence_text.lower()

    if any(pattern.search(sentence_lower) for pattern in EN_WH_MULTI_RE):
        return True

    for tok in sentence:
        lemma = tok.lemma_.lower()
        text = tok.text.lower()

        if tok.pos_ in {"PRON", "ADV", "DET", "SCONJ"} and lemma in EN_WH_SINGLE:
            return True

        if text in EN_WH_SINGLE:
            return True

    return False


def is_direct_wh_question(sentence):
    """Detect direct WH-questions using question mark + WH expression."""
    sentence_text = sentence.text.strip()

    if not sentence_text:
        return False

    return (
        sentence_ends_with_question_mark(sentence_text)
        and contains_wh_expression(sentence)
    )


# ============================================================
# Nominalisation detection
# ============================================================

def count_nominal_suffixes(chunk_tokens):
    """Count nouns whose lemma ends with a common English nominal suffix."""
    count = 0

    for tok in chunk_tokens:
        lemma = tok.lemma_.lower()

        if tok.pos_ == "NOUN" and lemma.endswith(EN_NOMINAL_SUFFIXES):
            count += 1

    return count


def count_gerund_nominals_ing(chunk_tokens):
    """
    Count nominal gerunds ending in -ing.

    spaCy often tags these as NOUN.
    """
    count = 0

    for tok in chunk_tokens:
        text = tok.text.lower()
        lemma = tok.lemma_.lower()

        if tok.pos_ == "NOUN" and (text.endswith("ing") or lemma.endswith("ing")):
            count += 1

    return count


# ============================================================
# Passive detection
# ============================================================

def sentence_has_by_phrase(sentence):
    """Check whether a sentence contains a by-phrase."""
    return any(
        tok.pos_ == "ADP" and tok.lemma_.lower() == "by"
        for tok in sentence
    )


def sentence_has_passive_pattern(sentence, window=4):
    """
    Approximate English passive constructions.

    Detects AUX with lemma 'be' or 'get' followed shortly by a participle:
    - was written
    - got injured
    - had been written
    """
    tokens = list(sentence)

    for i, tok in enumerate(tokens):
        lemma = tok.lemma_.lower()

        if tok.pos_ == "AUX" and lemma in AUX_PASSIVE_LEMMAS:
            j = i + 1
            steps = 0

            while j < len(tokens) and steps < window:
                candidate = tokens[j]

                if candidate.is_space:
                    j += 1
                    continue

                if (
                    candidate.pos_ in PASSIVE_SKIP_POS
                    or candidate.lemma_.lower() in NEGATION_LEMMAS
                ):
                    j += 1
                    steps += 1
                    continue

                if is_participle(candidate):
                    return True

                break

    return False


def classify_passive_sentence(sentence):
    """
    Classify a sentence as:
    - 'by' for by-passive
    - 'agentless' for passive without a by-phrase
    - None if no passive pattern is detected
    """
    if not sentence_has_passive_pattern(sentence):
        return None

    if sentence_has_by_phrase(sentence):
        return "by"

    return "agentless"


# ============================================================
# Book-level processing
# ============================================================

def process_book(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """
    Process one literary work.

    Sentence-level features are assigned to the 100-word chunk where the sentence starts.
    Nominalisation features are calculated directly over 100-word chunks.
    """
    text_batches = split_text_into_char_batches(text, batch_size=char_batch_size)

    global_word_index = 0
    all_word_tokens = []

    wh_question_positions = []
    by_passive_positions = []
    agentless_passive_positions = []

    for doc in nlp.pipe(text_batches, batch_size=1):
        for sentence in doc.sents:
            sentence_tokens = list(sentence)
            sentence_word_tokens = [tok for tok in sentence_tokens if is_word_token(tok)]
            sentence_word_count = len(sentence_word_tokens)

            if sentence_word_count == 0:
                continue

            sentence_start_word_index = global_word_index

            if is_direct_wh_question(sentence):
                wh_question_positions.append(sentence_start_word_index)

            passive_class = classify_passive_sentence(sentence)

            if passive_class == "by":
                by_passive_positions.append(sentence_start_word_index)
            elif passive_class == "agentless":
                agentless_passive_positions.append(sentence_start_word_index)

            all_word_tokens.extend(sentence_word_tokens)
            global_word_index += sentence_word_count

    token_total = global_word_index
    chunk_count = math.ceil(token_total / chunk_size) if token_total > 0 else 0

    wh_chunk_counts = [0] * chunk_count
    by_chunk_counts = [0] * chunk_count
    agentless_chunk_counts = [0] * chunk_count

    for position in wh_question_positions:
        wh_chunk_counts[position // chunk_size] += 1

    for position in by_passive_positions:
        by_chunk_counts[position // chunk_size] += 1

    for position in agentless_passive_positions:
        agentless_chunk_counts[position // chunk_size] += 1

    wh_values = []
    by_values = []
    agentless_values = []

    for i in range(chunk_count):
        current_chunk_size = get_last_chunk_size(
            token_total,
            chunk_index=i,
            chunk_count=chunk_count,
            chunk_size=chunk_size,
        )

        wh_values.append((wh_chunk_counts[i] / current_chunk_size) * 100.0)
        by_values.append((by_chunk_counts[i] / current_chunk_size) * 100.0)
        agentless_values.append((agentless_chunk_counts[i] / current_chunk_size) * 100.0)

    nominal_suffix_values = []
    gerund_nominal_values = []

    for chunk in split_into_word_chunks(all_word_tokens, chunk_size=chunk_size):
        if not chunk:
            continue

        current_chunk_size = len(chunk)

        nominal_suffix_values.append(
            (count_nominal_suffixes(chunk) / current_chunk_size) * 100.0
        )

        gerund_nominal_values.append(
            (count_gerund_nominals_ing(chunk) / current_chunk_size) * 100.0
        )

    result = {
        "chunk_size_words": chunk_size,
        "chunk_count_used": chunk_count,
        "token_total": token_total,
        "direct_wh_questions_total": len(wh_question_positions),
        "by_passives_total": len(by_passive_positions),
        "agentless_passives_total": len(agentless_passive_positions),
    }

    result.update(aggregate_feature(wh_values, "direct_wh_questions_per100"))
    result.update(aggregate_feature(nominal_suffix_values, "nominal_suffix_per100"))
    result.update(aggregate_feature(gerund_nominal_values, "gerund_nominals_ing_per100"))
    result.update(aggregate_feature(by_values, "by_passives_per100"))
    result.update(aggregate_feature(agentless_values, "agentless_passives_per100"))

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

        "direct_wh_questions_total",
        "direct_wh_questions_per100_avg",
        "direct_wh_questions_per100_min",
        "direct_wh_questions_per100_max",
        "direct_wh_questions_per100_sd",

        "nominal_suffix_per100_avg",
        "nominal_suffix_per100_min",
        "nominal_suffix_per100_max",
        "nominal_suffix_per100_sd",

        "gerund_nominals_ing_per100_avg",
        "gerund_nominals_ing_per100_min",
        "gerund_nominals_ing_per100_max",
        "gerund_nominals_ing_per100_sd",

        "by_passives_total",
        "by_passives_per100_avg",
        "by_passives_per100_min",
        "by_passives_per100_max",
        "by_passives_per100_sd",

        "agentless_passives_total",
        "agentless_passives_per100_avg",
        "agentless_passives_per100_min",
        "agentless_passives_per100_max",
        "agentless_passives_per100_sd",
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
        description="Extract English WH-question, nominalisation, and passive features."
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