"""
English sentence-fragment, dialogue, and indirect-speech features.

Features:
- no_verb_fragments_per100
- no_finite_fragments_per100
- dialogue_per100
- indirect_speech_per100

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
# Dialogue and speech patterns
# ============================================================

UPPERCASE_EN = r"A-Z"

QUOTE_PAIRS_LATIN = [
    ("«", "»"),
    ("“", "”"),
    ("„", "”"),
    ("‘", "’"),
    ("‚", "’"),
    ("‹", "›"),
    ('"', '"'),
]

RE_DIALOGUE_DASH = re.compile(rf"(?m)^[ \t]*[-—]{{1,2}}\s+[{UPPERCASE_EN}].*$")
RE_ANY_QUOTE = re.compile(r"[\"“”‘’«»‹›']")

EN_SPEECH_LEMMAS = {
    "say", "tell", "ask", "reply", "remark", "answer", "announce", "explain",
    "whisper", "shout", "yell", "murmur", "admit", "claim", "state", "suggest",
    "promise", "confess", "warn", "argue", "think", "believe", "remember", "note",
}


FEATURE_NAMES = [
    "no_verb_fragments_per100",
    "no_finite_fragments_per100",
    "dialogue_per100",
    "indirect_speech_per100",
]


def make_quoted_segment_regex(open_quote, close_quote):
    """Create regex for quoted dialogue beginning with a capital letter."""
    return re.compile(
        rf"{re.escape(open_quote)}\s*([{UPPERCASE_EN}].*?[\.?!…,])\s*{re.escape(close_quote)}",
        re.DOTALL,
    )


RE_DIALOGUE_QUOTES = [
    make_quoted_segment_regex(open_quote, close_quote)
    for open_quote, close_quote in QUOTE_PAIRS_LATIN
]


# ============================================================
# Helpers
# ============================================================

def load_spacy_model(model_name=SPACY_MODEL):
    """Load English spaCy model with rule-based sentence segmentation."""
    try:
        nlp = spacy.load(model_name, disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)

    nlp.max_length = 4_000_000
    return nlp


def is_word_token(tok):
    """Alphabetic word tokens only."""
    return tok.is_alpha


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
    """Return actual chunk size, accounting for final partial chunk."""
    if chunk_index < chunk_count - 1:
        return chunk_size

    final_size = token_total - (chunk_index * chunk_size)
    return final_size if final_size > 0 else chunk_size


def collect_regex_matches(regexes, text):
    """Collect all regex matches from a list of compiled patterns."""
    matches = []

    for regex in regexes:
        matches.extend(regex.finditer(text or ""))

    return matches


# ============================================================
# Sentence fragments
# ============================================================

def has_alpha_in_sentence(sentence):
    return any(tok.is_alpha for tok in sentence)


def classify_fragment_features(sentence):
    """
    Detect sentence fragments.

    no_verb_fragments_per100:
    - sentence contains no VERB or AUX

    no_finite_fragments_per100:
    - sentence contains no finite VERB or AUX
    """
    counts = {
        "no_verb_fragments_per100": 0,
        "no_finite_fragments_per100": 0,
    }

    if not has_alpha_in_sentence(sentence):
        return counts

    tokens = list(sentence)

    has_any_verb = any(tok.pos_ in {"VERB", "AUX"} for tok in tokens)
    has_finite_verb = any(
        tok.pos_ in {"VERB", "AUX"} and morph_has(tok, "VerbForm", "Fin")
        for tok in tokens
    )

    if not has_any_verb:
        counts["no_verb_fragments_per100"] = 1

    if not has_finite_verb:
        counts["no_finite_fragments_per100"] = 1

    return counts


# ============================================================
# Dialogue
# ============================================================

def find_dialogue_spans(text):
    """
    Detect direct dialogue spans using regexes only.

    This includes:
    - quoted dialogue beginning with a capital letter
    - dash-led dialogue lines
    """
    quote_matches = collect_regex_matches(RE_DIALOGUE_QUOTES, text or "")
    dash_matches = list(RE_DIALOGUE_DASH.finditer(text or ""))

    matches = quote_matches + dash_matches
    matches.sort(key=lambda match: match.start())

    return [(match.start(), match.end()) for match in matches]


def assign_dialogue_to_chunks(dialogue_spans, token_positions, chunk_size):
    """
    Assign each dialogue span to the chunk of the first word token at/after its start.
    """
    token_total = len(token_positions)
    chunk_count = math.ceil(token_total / chunk_size) if token_total > 0 else 0
    chunk_counts = [0] * chunk_count

    token_index = 0

    for start_char, _ in dialogue_spans:
        while token_index < token_total and token_positions[token_index] < start_char:
            token_index += 1

        if token_index >= token_total:
            continue

        chunk_index = token_index // chunk_size

        if 0 <= chunk_index < chunk_count:
            chunk_counts[chunk_index] += 1

    return chunk_counts


# ============================================================
# Indirect / reported speech
# ============================================================

def is_indirect_speech_sentence(sentence):
    """
    Approximate indirect/reported speech.

    A sentence is counted if it:
    - contains a speech/thought verb
    - does not contain obvious quote marks
    """
    sentence_text = sentence.text.strip()

    if RE_ANY_QUOTE.search(sentence_text):
        return False

    return any(
        tok.pos_ in {"VERB", "AUX"} and lemma(tok) in EN_SPEECH_LEMMAS
        for tok in sentence
    )


# ============================================================
# Book processing
# ============================================================

def process_book(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """Process one literary work and return book-level aggregate features."""
    text_batches = split_text_into_char_batches(text, batch_size=char_batch_size)

    dialogue_spans = find_dialogue_spans(text)

    global_word_index = 0
    token_positions = []

    feature_positions = {
        "no_verb_fragments_per100": [],
        "no_finite_fragments_per100": [],
        "indirect_speech_per100": [],
    }

    for doc in nlp.pipe(text_batches, batch_size=1):
        for tok in doc:
            if is_word_token(tok):
                token_positions.append(tok.idx)

        for sentence in doc.sents:
            sentence_tokens = list(sentence)
            sentence_word_count = sum(1 for tok in sentence_tokens if is_word_token(tok))

            if sentence_word_count == 0:
                continue

            sentence_start = global_word_index

            fragment_counts = classify_fragment_features(sentence)

            for feature_name, count in fragment_counts.items():
                if count > 0:
                    feature_positions[feature_name].extend([sentence_start] * count)

            if is_indirect_speech_sentence(sentence):
                feature_positions["indirect_speech_per100"].append(sentence_start)

            global_word_index += sentence_word_count

    token_total = global_word_index
    chunk_count = math.ceil(token_total / chunk_size) if token_total > 0 else 0

    dialogue_chunk_counts = assign_dialogue_to_chunks(
        dialogue_spans=dialogue_spans,
        token_positions=token_positions,
        chunk_size=chunk_size,
    )

    result = {
        "chunk_size_words": chunk_size,
        "chunk_count_used": chunk_count,
        "token_total": token_total,
        "dialogue_total": len(dialogue_spans),
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

    dialogue_values = []

    for i in range(chunk_count):
        current_chunk_size = get_current_chunk_size(
            token_total,
            chunk_index=i,
            chunk_count=chunk_count,
            chunk_size=chunk_size,
        )
        dialogue_count = dialogue_chunk_counts[i] if i < len(dialogue_chunk_counts) else 0
        dialogue_values.append((dialogue_count / current_chunk_size) * 100.0)

    result.update(aggregate_feature(dialogue_values, "dialogue_per100"))

    return result


# ============================================================
# Corpus processing
# ============================================================

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

        "no_verb_fragments_total",
        "no_verb_fragments_per100_avg",
        "no_verb_fragments_per100_min",
        "no_verb_fragments_per100_max",
        "no_verb_fragments_per100_sd",

        "no_finite_fragments_total",
        "no_finite_fragments_per100_avg",
        "no_finite_fragments_per100_min",
        "no_finite_fragments_per100_max",
        "no_finite_fragments_per100_sd",

        "dialogue_total",
        "dialogue_per100_avg",
        "dialogue_per100_min",
        "dialogue_per100_max",
        "dialogue_per100_sd",

        "indirect_speech_total",
        "indirect_speech_per100_avg",
        "indirect_speech_per100_min",
        "indirect_speech_per100_max",
        "indirect_speech_per100_sd",
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
        description="Extract English sentence-fragment, dialogue, and indirect-speech features."
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