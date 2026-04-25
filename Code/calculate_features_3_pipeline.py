"""
English stative, subordination, and participial-clause features.

Features:
- copular_per100
- existential_per100
- that_verb_ccomp_per100
- that_adj_ccomp_per100
- wh_free_rel_per100
- infinitive_per100
- present_participial_per100
- past_participial_per100

The user should provide:
- an input folder containing cleaned .txt files
- an output CSV path
"""

import argparse
import glob
import math
from pathlib import Path

import pandas as pd
import spacy


CHUNK_SIZE = 100
CHAR_BATCH_SIZE = 100_000
SPACY_MODEL = "en_core_web_sm"

THAT_MARKERS = {"that"}
WH_SET = {
    "what", "whatever", "who", "whoever", "whom",
    "which", "where", "when", "why", "how"
}


# ============================================================
# General helpers
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


def morph_has(tok, key, value):
    values = tok.morph.get(key)
    return isinstance(values, list) and value in values


def has_finite_verb(tokens):
    return any(
        tok.pos_ in {"VERB", "AUX"} and morph_has(tok, "VerbForm", "Fin")
        for tok in tokens
    )


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
    """Return chunk size, accounting for the final partial chunk."""
    if chunk_index < chunk_count - 1:
        return chunk_size

    final_size = token_total - (chunk_index * chunk_size)
    return final_size if final_size > 0 else chunk_size


# ============================================================
# Stative features
# ============================================================

def is_existential_sentence(sentence):
    """Detect existential there + be constructions."""
    lemmas = [tok.lemma_.lower() for tok in sentence]
    return "there" in lemmas and "be" in lemmas


def is_copular_sentence(sentence):
    """
    Detect copular constructions.

    Preferred signal: dependency label cop.
    Fallback: sentence contains be and is not existential.
    """
    for tok in sentence:
        if tok.lemma_.lower() == "be" and tok.dep_ == "cop":
            return True

    lemmas = [tok.lemma_.lower() for tok in sentence]

    if "be" in lemmas and not is_existential_sentence(sentence):
        return True

    return False


# ============================================================
# Subordination features
# ============================================================

def sentence_that_verb_ccomp(sentence):
    """Detect verb head with ccomp child marked by that."""
    for head in sentence:
        if head.pos_ != "VERB":
            continue

        for child in head.children:
            if child.dep_ == "ccomp":
                if any(
                    tok.dep_ == "mark" and tok.lemma_.lower() in THAT_MARKERS
                    for tok in child.subtree
                ):
                    return True

    return False


def sentence_that_adj_ccomp(sentence):
    """Detect adjective predicate taking a finite that-clause complement."""
    for adj in sentence:
        if adj.pos_ != "ADJ":
            continue

        for child in adj.children:
            if child.dep_ == "ccomp":
                subtree = list(child.subtree)
                if any(
                    tok.dep_ == "mark" and tok.lemma_.lower() in THAT_MARKERS
                    for tok in subtree
                ) and has_finite_verb(subtree):
                    return True

        subtree = list(adj.subtree)

        if any(
            tok.dep_ == "mark" and tok.lemma_.lower() in THAT_MARKERS
            for tok in subtree
        ) and has_finite_verb(subtree):
            return True

        for marker in sentence:
            if marker.dep_ != "mark" or marker.lemma_.lower() not in THAT_MARKERS:
                continue

            head = marker.head
            current = head
            dominated_by_adj = False

            while current is not None and current != current.head:
                if current == adj:
                    dominated_by_adj = True
                    break
                current = current.head

            if dominated_by_adj and has_finite_verb(head.subtree):
                return True

    return False


def sentence_wh_free_rel(sentence):
    """Detect WH item inside or heading a finite clause."""
    for tok in sentence:
        if tok.lemma_.lower() not in WH_SET:
            continue

        clause_head = tok

        while (
            clause_head.dep_ not in {"ccomp", "xcomp", "advcl", "acl", "relcl"}
            and clause_head != clause_head.head
        ):
            clause_head = clause_head.head

        subtree = clause_head.subtree if clause_head != clause_head.head else tok.subtree

        if has_finite_verb(subtree):
            return True

    return False


def sentence_infinitive(sentence):
    """Detect any infinitive verb form in the sentence."""
    return any(morph_has(tok, "VerbForm", "Inf") for tok in sentence)


# ============================================================
# Participial-clause features
# ============================================================

def looks_detached_with_comma(tok):
    """Detect sentence-initial detached participial phrases followed by comma."""
    sentence_tokens = list(tok.sent)
    relative_index = tok.i - sentence_tokens[0].i

    if relative_index <= 2:
        for j in range(relative_index + 1, min(relative_index + 6, len(sentence_tokens))):
            if sentence_tokens[j].text == ",":
                return True

    return False


def is_english_present_participle(tok):
    return morph_has(tok, "VerbForm", "Part") and tok.text.lower().endswith("ing")


def is_english_past_participle(tok):
    return morph_has(tok, "VerbForm", "Part") and not tok.text.lower().endswith("ing")


def looks_participial_clause_context(tok):
    """Detect clause-like contexts for participial constructions."""
    return tok.dep_ in {"advcl", "acl"} or looks_detached_with_comma(tok)


def sentence_participial_features(sentence):
    """Return whether present/past participial clauses occur in the sentence."""
    has_present = False
    has_past = False

    for tok in sentence:
        is_present = is_english_present_participle(tok)
        is_past = is_english_past_participle(tok)

        if not (is_present or is_past):
            continue

        if not looks_participial_clause_context(tok):
            continue

        if is_present:
            has_present = True

        if is_past:
            has_past = True

    return has_present, has_past


# ============================================================
# Book processing
# ============================================================

FEATURE_NAMES = [
    "copular_per100",
    "existential_per100",
    "that_verb_ccomp_per100",
    "that_adj_ccomp_per100",
    "wh_free_rel_per100",
    "infinitive_per100",
    "present_participial_per100",
    "past_participial_per100",
]


def process_book(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """
    Process one literary work.

    Each sentence-level feature is assigned to the 100-word chunk where
    the sentence begins.
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

            if is_copular_sentence(sentence):
                feature_positions["copular_per100"].append(sentence_start)

            if is_existential_sentence(sentence):
                feature_positions["existential_per100"].append(sentence_start)

            if sentence_that_verb_ccomp(sentence):
                feature_positions["that_verb_ccomp_per100"].append(sentence_start)

            if sentence_that_adj_ccomp(sentence):
                feature_positions["that_adj_ccomp_per100"].append(sentence_start)

            if sentence_wh_free_rel(sentence):
                feature_positions["wh_free_rel_per100"].append(sentence_start)

            if sentence_infinitive(sentence):
                feature_positions["infinitive_per100"].append(sentence_start)

            has_present_part, has_past_part = sentence_participial_features(sentence)

            if has_present_part:
                feature_positions["present_participial_per100"].append(sentence_start)

            if has_past_part:
                feature_positions["past_participial_per100"].append(sentence_start)

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
        description="Extract English stative, subordination, and participial-clause features."
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