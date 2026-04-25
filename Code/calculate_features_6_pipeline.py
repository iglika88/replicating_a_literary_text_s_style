"""
English specialized verb, coordination, and negation features.

Features:
- public_reporting_per100
- private_cognition_per100
- suasive_directive_per100
- seem_appear_per100
- phrasal_coord_per100
- clause_coord_per100
- analytic_negation_per100
- synthetic_negation_per100

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
CHAR_BATCH_SIZE = 200_000
SPACY_MODEL = "en_core_web_sm"


# ============================================================
# Inventories
# ============================================================

EN_PUBLIC = {
    "say", "declare", "announce", "state", "explain", "mention", "reply",
    "add", "recount", "admit", "acknowledge", "proclaim", "maintain", "claim",
}

EN_PRIVATE = {
    "think", "believe", "know", "doubt", "suppose", "imagine", "estimate",
    "understand", "remember", "forget", "hope", "expect", "fear",
}

EN_SUASIVE = {
    "order", "command", "demand", "ask", "propose", "suggest", "recommend",
    "advise", "urge", "forbid", "permit", "enjoin", "insist",
}

EN_SEEM = {"seem", "appear", "look"}
RE_EN_LOOK_LIKE = re.compile(r"\blook(s|ed|ing)?\s+like\b", re.IGNORECASE)

CLAUSE_INIT_SET = {"and", "but", "or", "so", "then", "also", "plus"}
START_REGEXES = [re.compile(r"^\s*(and|but|or|so|then|also|plus)\b", re.IGNORECASE)]
SHARED_POS_FOR_PHRASAL = {"NOUN", "ADJ", "VERB", "ADV"}

EN_ANALYTIC_NEG_LEMMAS = {"not", "n't"}
RE_NEITHER_NOR = re.compile(r"\bneither\b.*\bnor\b", re.IGNORECASE)
RE_WITHOUT = re.compile(r"\bwithout\b", re.IGNORECASE)


FEATURE_NAMES = [
    "public_reporting_per100",
    "private_cognition_per100",
    "suasive_directive_per100",
    "seem_appear_per100",
    "phrasal_coord_per100",
    "clause_coord_per100",
    "analytic_negation_per100",
    "synthetic_negation_per100",
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


def lemma(tok):
    return (tok.lemma_ or tok.text).lower()


def morph_has(tok, key, value):
    values = tok.morph.get(key)
    return isinstance(values, list) and value in values


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
# Specialized verb classes
# ============================================================

def is_verbish(tok):
    return tok.pos_ in {"VERB", "AUX"}


def classify_specialized_verbs(sentence):
    """Count public, private, suasive, and seem/appear verbs in one sentence."""
    counts = {
        "public_reporting_per100": 0,
        "private_cognition_per100": 0,
        "suasive_directive_per100": 0,
        "seem_appear_per100": 0,
    }

    sentence_text = sentence.text.strip().lower()

    for tok in sentence:
        if not is_verbish(tok):
            continue

        tok_lemma = lemma(tok)

        if tok_lemma in EN_PUBLIC:
            counts["public_reporting_per100"] += 1

        if tok_lemma in EN_PRIVATE:
            counts["private_cognition_per100"] += 1

        if tok_lemma in EN_SUASIVE:
            counts["suasive_directive_per100"] += 1

        if tok_lemma in EN_SEEM:
            counts["seem_appear_per100"] += 1

    if RE_EN_LOOK_LIKE.search(sentence_text):
        counts["seem_appear_per100"] += 1

    return counts


# ============================================================
# Coordination
# ============================================================

def classify_coordination(sentence):
    """
    Count phrasal and clause coordination.

    phrasal_coord:
    - UD conj edges where head and child share POS in selected POS categories

    clause_coord:
    - sentence-initial coordinator/discourse connector + nearby finite verb
    """
    counts = {
        "phrasal_coord_per100": 0,
        "clause_coord_per100": 0,
    }

    sentence_text = sentence.text.strip().lower()

    for tok in sentence:
        if tok.dep_ == "conj":
            if tok.pos_ == tok.head.pos_ and tok.pos_ in SHARED_POS_FOR_PHRASAL:
                counts["phrasal_coord_per100"] += 1

    sentence_tokens = [tok for tok in sentence if not tok.is_space]
    ok_start = False

    if sentence_tokens:
        first_nonpunct = None

        for tok in sentence_tokens:
            if not tok.is_punct:
                first_nonpunct = tok
                break

        if first_nonpunct is not None:
            if first_nonpunct.lower_ in CLAUSE_INIT_SET:
                ok_start = True
            elif any(pattern.match(sentence_text) for pattern in START_REGEXES):
                ok_start = True

    if ok_start:
        nearby = sentence_tokens[:7]
        has_finite_verb = any(
            tok.pos_ in {"VERB", "AUX"} and morph_has(tok, "VerbForm", "Fin")
            for tok in nearby
        )

        if has_finite_verb:
            counts["clause_coord_per100"] = 1

    return counts


# ============================================================
# Negation
# ============================================================

def has_dep_neg(sentence):
    return any(tok.dep_ == "neg" or tok.dep_.endswith(":neg") for tok in sentence)


def has_analytic_negation(sentence):
    """Detect analytic negation: neg dependency or not/n't."""
    if has_dep_neg(sentence):
        return True

    for tok in sentence:
        tok_lemma = lemma(tok)
        tok_text = tok.text.lower()

        if tok_lemma in EN_ANALYTIC_NEG_LEMMAS or tok_text in EN_ANALYTIC_NEG_LEMMAS:
            return True

    return False


def has_synthetic_negation(sentence):
    """
    Detect synthetic negation:
    - nobody, no one, nothing, none, nowhere, neither
    - neither ... nor
    - without

    Synthetic negation is suppressed if analytic negation is already present.
    """
    if has_analytic_negation(sentence):
        return False

    sentence_text = sentence.text.strip().lower()

    for tok in sentence:
        tok_text = tok.text.lower()
        tok_lemma = lemma(tok)

        if tok_text in {"nobody", "nothing", "none", "nowhere", "neither"}:
            return True

        if tok_lemma in {"nobody", "nothing", "none", "nowhere", "neither"}:
            return True

    if "no one" in sentence_text:
        return True

    if RE_NEITHER_NOR.search(sentence_text):
        return True

    if RE_WITHOUT.search(sentence_text):
        return True

    return False


def classify_negation(sentence):
    """Count analytic and synthetic negation, at most once per sentence."""
    counts = {
        "analytic_negation_per100": 0,
        "synthetic_negation_per100": 0,
    }

    if has_analytic_negation(sentence):
        counts["analytic_negation_per100"] = 1
        return counts

    if has_synthetic_negation(sentence):
        counts["synthetic_negation_per100"] = 1

    return counts


# ============================================================
# Book processing
# ============================================================

def process_book(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """Process one literary work and return book-level aggregate features."""
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

            combined_counts = {}
            combined_counts.update(classify_specialized_verbs(sentence))
            combined_counts.update(classify_coordination(sentence))
            combined_counts.update(classify_negation(sentence))

            for feature_name, count in combined_counts.items():
                if count > 0:
                    feature_positions[feature_name].extend([sentence_start] * count)

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
        description="Extract English specialized verb, coordination, and negation features."
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