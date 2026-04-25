"""
English relative, adverbial-subordination, adjective, adverb, and prepositional features.

Features:
- rel_sentence_per100
- causal_subord_per100
- concessive_subord_per100
- conditional_subord_per100
- other_subord_per100
- total_pps_per100
- attributive_adjs_per100
- predicative_adjs_per100
- total_adverbs_per100

The user should provide:
- an input folder containing cleaned .txt files
- an output CSV path
"""

import argparse
import glob
import math
import re
from pathlib import Path

import pandas as pd
import spacy


CHUNK_SIZE = 100
CHAR_BATCH_SIZE = 100_000
SPACY_MODEL = "en_core_web_sm"


# ============================================================
# Inventories and regexes
# ============================================================

SENTENCE_RELATIVE_RE = re.compile(r",\s*which\b", re.IGNORECASE)

UD_MARK_LEMMA_BUCKETS_EN = {
    "because": "causal_subord_per100",
    "since": "causal_subord_per100",

    "although": "concessive_subord_per100",
    "though": "concessive_subord_per100",
    "whereas": "concessive_subord_per100",

    "if": "conditional_subord_per100",
    "unless": "conditional_subord_per100",
    "provided": "conditional_subord_per100",
    "providing": "conditional_subord_per100",

    "when": "other_subord_per100",
    "while": "other_subord_per100",
    "after": "other_subord_per100",
    "before": "other_subord_per100",
    "until": "other_subord_per100",
    "once": "other_subord_per100",
    "lest": "other_subord_per100",
}

MWE_EN = [
    (re.compile(r"\bbecause\b", re.IGNORECASE), "causal_subord_per100"),
    (re.compile(r"\bsince\b", re.IGNORECASE), "causal_subord_per100"),
    (re.compile(r"\bas\b", re.IGNORECASE), "causal_subord_per100"),
    (re.compile(r"\bnow that\b", re.IGNORECASE), "causal_subord_per100"),
    (re.compile(r"\bseeing that\b", re.IGNORECASE), "causal_subord_per100"),

    (re.compile(r"\balthough\b", re.IGNORECASE), "concessive_subord_per100"),
    (re.compile(r"\beven though\b", re.IGNORECASE), "concessive_subord_per100"),
    (re.compile(r"\bthough\b", re.IGNORECASE), "concessive_subord_per100"),
    (re.compile(r"\bwhereas\b", re.IGNORECASE), "concessive_subord_per100"),
    (re.compile(r"\beven if\b", re.IGNORECASE), "concessive_subord_per100"),

    (re.compile(r"\bif\b", re.IGNORECASE), "conditional_subord_per100"),
    (re.compile(r"\bunless\b", re.IGNORECASE), "conditional_subord_per100"),
    (re.compile(r"\bprovided that\b", re.IGNORECASE), "conditional_subord_per100"),
    (re.compile(r"\bproviding(?:\s+that)?\b", re.IGNORECASE), "conditional_subord_per100"),
    (re.compile(r"\bas long as\b", re.IGNORECASE), "conditional_subord_per100"),
    (re.compile(r"\bin case\b", re.IGNORECASE), "conditional_subord_per100"),

    (re.compile(r"\bwhen\b", re.IGNORECASE), "other_subord_per100"),
    (re.compile(r"\bwhile\b", re.IGNORECASE), "other_subord_per100"),
    (re.compile(r"\bafter\b", re.IGNORECASE), "other_subord_per100"),
    (re.compile(r"\bbefore\b", re.IGNORECASE), "other_subord_per100"),
    (re.compile(r"\buntil\b", re.IGNORECASE), "other_subord_per100"),
    (re.compile(r"\bonce\b", re.IGNORECASE), "other_subord_per100"),
    (re.compile(r"\bso that\b", re.IGNORECASE), "other_subord_per100"),
    (re.compile(r"\bin order that\b", re.IGNORECASE), "other_subord_per100"),
    (re.compile(r"\bas soon as\b", re.IGNORECASE), "other_subord_per100"),
]


FEATURE_NAMES = [
    "rel_sentence_per100",
    "causal_subord_per100",
    "concessive_subord_per100",
    "conditional_subord_per100",
    "other_subord_per100",
    "total_pps_per100",
    "attributive_adjs_per100",
    "predicative_adjs_per100",
    "total_adverbs_per100",
]


# ============================================================
# Helpers
# ============================================================

def load_spacy_model(model_name=SPACY_MODEL):
    """Load English spaCy model with parser enabled."""
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
    return not tok.is_space and not tok.is_punct


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


def get_current_chunk_size(token_total, chunk_index, chunk_count, chunk_size):
    if chunk_index < chunk_count - 1:
        return chunk_size

    final_size = token_total - (chunk_index * chunk_size)
    return final_size if final_size > 0 else chunk_size


# ============================================================
# Sentence-level feature detection
# ============================================================

def sentence_has_sentence_relative(sentence):
    """
    Detect sentence relatives of the type ', which ...'.

    Only this relative-clause feature is retained, because the other
    relative-clause categories were not efficient enough for the current study.
    """
    return bool(SENTENCE_RELATIVE_RE.search(sentence.text.lower()))


def classify_adverbial_subordinators(sentence):
    """
    Detect broad categories of adverbial subordinators.

    A sentence may be counted in more than one category.
    Each category is counted at most once per sentence.
    """
    flags = {
        "causal_subord_per100": False,
        "concessive_subord_per100": False,
        "conditional_subord_per100": False,
        "other_subord_per100": False,
    }

    sentence_text = sentence.text.lower()

    for tok in sentence:
        if tok.dep_ == "mark":
            lemma = tok.lemma_.lower()
            if lemma in UD_MARK_LEMMA_BUCKETS_EN:
                flags[UD_MARK_LEMMA_BUCKETS_EN[lemma]] = True

    for pattern, feature_name in MWE_EN:
        if pattern.search(sentence_text):
            flags[feature_name] = True

    return flags


def find_token_level_features_in_sentence(sentence, sentence_start_word_index):
    """
    Return absolute word positions for:
    - prepositional phrases
    - attributive adjectives
    - predicative adjectives
    - adverbs
    """
    positions = {
        "total_pps_per100": [],
        "attributive_adjs_per100": [],
        "predicative_adjs_per100": [],
        "total_adverbs_per100": [],
    }

    word_pos_in_sentence = 0

    for tok in sentence:
        if not is_word_token(tok):
            continue

        absolute_word_position = sentence_start_word_index + word_pos_in_sentence

        if tok.pos_ == "ADP" and tok.dep_ == "prep":
            positions["total_pps_per100"].append(absolute_word_position)

        if tok.pos_ == "ADJ" and tok.dep_ == "amod" and tok.head.pos_ in {"NOUN", "PROPN"}:
            positions["attributive_adjs_per100"].append(absolute_word_position)

        if tok.pos_ == "ADJ" and tok.dep_ == "acomp":
            positions["predicative_adjs_per100"].append(absolute_word_position)
        elif tok.pos_ == "ADJ" and any(child.dep_ == "cop" for child in tok.children):
            positions["predicative_adjs_per100"].append(absolute_word_position)

        if tok.pos_ == "ADV":
            positions["total_adverbs_per100"].append(absolute_word_position)

        word_pos_in_sentence += 1

    return positions


# ============================================================
# Book processing
# ============================================================

def process_book(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """
    Process one literary work.

    Sentence-level features are assigned to the chunk where the sentence starts.
    Token-level features are assigned to the chunk where the token occurs.
    """
    text_batches = split_text_into_char_batches(text, batch_size=char_batch_size)

    global_word_index = 0
    feature_positions = {name: [] for name in FEATURE_NAMES}

    for doc in nlp.pipe(text_batches, batch_size=1):
        for sentence in doc.sents:
            sentence_tokens = list(sentence)
            sentence_word_count = sum(1 for tok in sentence_tokens if is_word_token(tok))

            if sentence_word_count == 0:
                continue

            sentence_start = global_word_index

            if sentence_has_sentence_relative(sentence):
                feature_positions["rel_sentence_per100"].append(sentence_start)

            subord_flags = classify_adverbial_subordinators(sentence)

            for feature_name, is_present in subord_flags.items():
                if is_present:
                    feature_positions[feature_name].append(sentence_start)

            token_positions = find_token_level_features_in_sentence(
                sentence,
                sentence_start_word_index=sentence_start,
            )

            for feature_name, positions in token_positions.items():
                feature_positions[feature_name].extend(positions)

            global_word_index += sentence_word_count

    token_total = global_word_index
    chunk_count = math.ceil(token_total / chunk_size) if token_total > 0 else 0

    result = {
        "chunk_size_words": chunk_size,
        "chunk_count_used": chunk_count,
        "token_total": token_total,
    }

    for feature_name, positions in feature_positions.items():
        total_name = feature_name.replace("_per100", "_total")
        result[total_name] = len(positions)

        chunk_counts = [0] * chunk_count

        for position in positions:
            chunk_index = position // chunk_size
            if 0 <= chunk_index < chunk_count:
                chunk_counts[chunk_index] += 1

        values = []

        for i in range(chunk_count):
            current_chunk_size = get_current_chunk_size(
                token_total,
                chunk_index=i,
                chunk_count=chunk_count,
                chunk_size=chunk_size,
            )
            values.append((chunk_counts[i] / current_chunk_size) * 100.0)

        result.update(aggregate_feature(values, feature_name))

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
    ]

    for feature in FEATURE_NAMES:
        total_name = feature.replace("_per100", "_total")
        row_order.extend([
            total_name,
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
        description=(
            "Extract English sentence-relative, adverbial-subordination, "
            "prepositional, adjective, and adverb features."
        )
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