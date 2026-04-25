"""
English content-suitability, first-person narration, and first-sentence features.

Chunk-based features:
- death_words_per100
- sex_words_per100
- total_flagged_words_per100
- first_person_narration_per100

Book-level first-sentence features:
- n_words_first_sentence
- n_chars_first_sentence

The user should provide:
- an input folder containing cleaned .txt files
- an output CSV path for chunk-based numeric features
- an output CSV path for first-sentence numeric features
- optionally, an output CSV path for first-sentence text
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
# Lexicons
# ============================================================

DEATH_LEMMAS = {
    "die", "dead", "death", "kill", "murder", "homicide", "suicide",
    "slay", "slain", "slaughter", "execute", "assassinate",
}

SEX_LEMMAS = {
    "sex", "sexual", "sexuality", "penis", "vagina", "breast", "breasts",
    "dick", "cock", "pussy", "fuck", "rape",
}

FIRST_PERSON_LEMMAS = {"i", "me", "we", "us", "my", "our", "mine", "ours"}


# ============================================================
# Quotation/dialogue patterns
# ============================================================

QUOTE_PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ("«", "»"),
    ("“", "”"),
    ("‘", "’"),
]

DASH_LINE_RE = re.compile(r"(?m)^[ \t]*[—–-][ \t].*$")

SENTENCE_ENDERS = {".", "!", "?", "…"}
SENTENCE_CLOSERS = {'"', "»", "”", ")", "]", "'", "›"}


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

    nlp.max_length = 3_000_000
    return nlp


def is_word_token(tok):
    """Alphabetic word tokens; excludes spaces, punctuation, symbols, and numbers."""
    if tok.is_space or tok.is_punct or tok.like_num or tok.pos_ == "SYM":
        return False
    return any(ch.isalpha() for ch in tok.text)


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
# Non-child-friendly word features
# ============================================================

def compute_non_child_friendly_features(chunk_tokens):
    """Count death-related and sex-related lemmas in one chunk."""
    n_words = len(chunk_tokens)

    if n_words == 0:
        return {
            "death_words_per100": 0.0,
            "sex_words_per100": 0.0,
            "total_flagged_words_per100": 0.0,
            "death_words_total": 0,
            "sex_words_total": 0,
            "total_flagged_words_total": 0,
        }

    death_count = 0
    sex_count = 0

    for tok in chunk_tokens:
        tok_lemma = lemma(tok)

        if tok_lemma in DEATH_LEMMAS:
            death_count += 1
        elif tok_lemma in SEX_LEMMAS:
            sex_count += 1

    total_count = death_count + sex_count
    scale = 100.0 / n_words

    return {
        "death_words_per100": death_count * scale,
        "sex_words_per100": sex_count * scale,
        "total_flagged_words_per100": total_count * scale,
        "death_words_total": death_count,
        "sex_words_total": sex_count,
        "total_flagged_words_total": total_count,
    }


# ============================================================
# First-person narration outside direct speech
# ============================================================

def find_quote_spans(text):
    """Find quoted or dash-led dialogue spans in the original text."""
    spans = []

    for opener, closer in QUOTE_PAIRS:
        pattern = re.compile(
            re.escape(opener) + r"(.*?)" + re.escape(closer),
            re.DOTALL,
        )
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text or ""))

    for match in DASH_LINE_RE.finditer(text or ""):
        spans.append((match.start(), match.end()))

    if not spans:
        return []

    spans.sort()
    merged = [spans[0]]

    for start, end in spans[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def in_spans(index, spans):
    """Check whether a character index falls inside any span."""
    return any(start <= index < end for start, end in spans)


def line_starts_with_dash(text, index):
    """Check whether the current line is dash-led dialogue."""
    line_start = text.rfind("\n", 0, index) + 1
    line_end = text.find("\n", index)

    if line_end == -1:
        line_end = len(text)

    line = text[line_start:line_end]
    return bool(re.match(r"^[ \t]*[—–-][ \t]", line))


def count_first_person_narration(sentence, full_text, speech_spans):
    """
    Count first-person pronouns/determiners outside quoted or dash-led speech.
    """
    count = 0

    for tok in sentence:
        if tok.pos_ not in {"PRON", "DET"}:
            continue

        if lemma(tok) not in FIRST_PERSON_LEMMAS:
            continue

        if in_spans(tok.idx, speech_spans):
            continue

        if line_starts_with_dash(full_text, tok.idx):
            continue

        count += 1

    return count


# ============================================================
# First-sentence features
# ============================================================

def extract_first_sentence_fast(text, max_scan_chars=50_000):
    """
    Extract the first likely sentence without sending the whole novel to spaCy.
    """
    text = (text or "").strip()

    if not text:
        return ""

    snippet = text[:max_scan_chars]
    n = len(snippet)
    i = 0

    while i < n:
        char = snippet[i]

        if char in SENTENCE_ENDERS:
            j = i + 1

            while j < n and snippet[j] in SENTENCE_CLOSERS:
                j += 1

            if j >= n or snippet[j].isspace():
                candidate = snippet[:j].strip()

                if candidate:
                    return candidate

        i += 1

    first_paragraph = re.split(r"\n\s*\n|\n", snippet)[0].strip()
    return first_paragraph if first_paragraph else snippet.strip()


def extract_first_sentence_features(text, nlp):
    """Extract first sentence and compute word/character length."""
    first_sentence_text = extract_first_sentence_fast(text)

    if not first_sentence_text:
        return {
            "first_sentence": "",
            "n_words_first_sentence": 0,
            "n_chars_first_sentence": 0,
        }

    sentence_doc = nlp(first_sentence_text)

    final_sentence = ""

    for sentence in sentence_doc.sents:
        if any(tok.is_alpha for tok in sentence):
            final_sentence = sentence.text.strip()
            break

    if not final_sentence:
        return {
            "first_sentence": "",
            "n_words_first_sentence": 0,
            "n_chars_first_sentence": 0,
        }

    final_doc = nlp(final_sentence)
    n_words = sum(1 for tok in final_doc if tok.is_alpha)

    return {
        "first_sentence": final_sentence,
        "n_words_first_sentence": n_words,
        "n_chars_first_sentence": len(final_sentence),
    }


# ============================================================
# Book processing
# ============================================================

def process_book(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """Process chunk-based content and narration features for one book."""
    speech_spans = find_quote_spans(text)
    text_batches = split_text_into_char_batches(text, batch_size=char_batch_size)

    word_tokens = []
    first_person_positions = []
    global_word_index = 0

    for doc in nlp.pipe(text_batches, batch_size=1):
        for sentence in doc.sents:
            sentence_tokens = list(sentence)
            sentence_word_tokens = [tok for tok in sentence_tokens if is_word_token(tok)]
            sentence_word_count = len(sentence_word_tokens)

            if sentence_word_count == 0:
                continue

            sentence_start = global_word_index

            first_person_count = count_first_person_narration(
                sentence=sentence,
                full_text=text,
                speech_spans=speech_spans,
            )

            if first_person_count > 0:
                first_person_positions.extend([sentence_start] * first_person_count)

            word_tokens.extend(sentence_word_tokens)
            global_word_index += sentence_word_count

    chunks = split_into_chunks(word_tokens, chunk_size=chunk_size)

    collected = {
        "death_words_per100": [],
        "sex_words_per100": [],
        "total_flagged_words_per100": [],
        "first_person_narration_per100": [],
    }

    totals = {
        "death_words_total": 0,
        "sex_words_total": 0,
        "total_flagged_words_total": 0,
        "first_person_narration_total": len(first_person_positions),
    }

    token_total = len(word_tokens)
    chunk_count = len(chunks)

    first_person_chunk_counts = [0] * chunk_count

    for position in first_person_positions:
        chunk_index = position // chunk_size

        if 0 <= chunk_index < chunk_count:
            first_person_chunk_counts[chunk_index] += 1

    for i, chunk in enumerate(chunks):
        suitability = compute_non_child_friendly_features(chunk)

        for feature in [
            "death_words_per100",
            "sex_words_per100",
            "total_flagged_words_per100",
        ]:
            collected[feature].append(suitability[feature])

        totals["death_words_total"] += suitability["death_words_total"]
        totals["sex_words_total"] += suitability["sex_words_total"]
        totals["total_flagged_words_total"] += suitability["total_flagged_words_total"]

        current_chunk_size = len(chunk) if chunk else chunk_size
        first_person_value = (first_person_chunk_counts[i] / current_chunk_size) * 100.0
        collected["first_person_narration_per100"].append(first_person_value)

    result = {
        "chunk_size_words": chunk_size,
        "chunk_count_used": chunk_count,
        "token_total": token_total,
        **totals,
    }

    for feature, values in collected.items():
        result.update(aggregate_feature(values, feature))

    return result


# ============================================================
# Corpus processing
# ============================================================

def process_corpus(
    input_dir,
    output_csv,
    first_sentence_numeric_csv,
    first_sentence_text_csv=None,
    file_pattern="*_cleaned.txt",
):
    """Process all cleaned text files and save chunk-based and first-sentence outputs."""
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    first_sentence_numeric_csv = Path(first_sentence_numeric_csv)

    text_files = sorted(glob.glob(str(input_dir / file_pattern)))

    if not text_files:
        raise RuntimeError(f"No files matching '{file_pattern}' were found in: {input_dir}")

    nlp = load_spacy_model()

    chunk_results = {}
    first_sentence_numeric = {}
    first_sentence_text_rows = []

    print(f"Found {len(text_files)} text files.")

    for idx, filepath in enumerate(text_files, start=1):
        filepath = Path(filepath)
        work_name = filepath.name.replace("_cleaned.txt", "")

        print(f"[{idx}/{len(text_files)}] Processing: {work_name}")

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

        chunk_results[work_name] = process_book(text, nlp=nlp)

        first_sentence = extract_first_sentence_features(text, nlp=nlp)

        first_sentence_numeric[work_name] = {
            "n_words_first_sentence": first_sentence["n_words_first_sentence"],
            "n_chars_first_sentence": first_sentence["n_chars_first_sentence"],
        }

        first_sentence_text_rows.append({
            "text_name": work_name,
            "first_sentence": first_sentence["first_sentence"],
            "n_words_first_sentence": first_sentence["n_words_first_sentence"],
            "n_chars_first_sentence": first_sentence["n_chars_first_sentence"],
        })

    chunk_df = pd.DataFrame(chunk_results)

    chunk_row_order = [
        "chunk_size_words",
        "chunk_count_used",
        "token_total",

        "death_words_total",
        "death_words_per100_avg",
        "death_words_per100_min",
        "death_words_per100_max",
        "death_words_per100_sd",

        "sex_words_total",
        "sex_words_per100_avg",
        "sex_words_per100_min",
        "sex_words_per100_max",
        "sex_words_per100_sd",

        "total_flagged_words_total",
        "total_flagged_words_per100_avg",
        "total_flagged_words_per100_min",
        "total_flagged_words_per100_max",
        "total_flagged_words_per100_sd",

        "first_person_narration_total",
        "first_person_narration_per100_avg",
        "first_person_narration_per100_min",
        "first_person_narration_per100_max",
        "first_person_narration_per100_sd",
    ]

    chunk_df = chunk_df.loc[chunk_row_order]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    chunk_df.to_csv(output_csv, encoding="utf-8")

    first_sentence_numeric_df = pd.DataFrame(first_sentence_numeric)
    first_sentence_numeric_df = first_sentence_numeric_df.loc[
        ["n_words_first_sentence", "n_chars_first_sentence"]
    ]

    first_sentence_numeric_csv.parent.mkdir(parents=True, exist_ok=True)
    first_sentence_numeric_df.to_csv(first_sentence_numeric_csv, encoding="utf-8")

    if first_sentence_text_csv is not None:
        first_sentence_text_csv = Path(first_sentence_text_csv)
        first_sentence_text_csv.parent.mkdir(parents=True, exist_ok=True)

        first_sentence_text_df = (
            pd.DataFrame(first_sentence_text_rows)
            .sort_values("text_name")
        )
        first_sentence_text_df.to_csv(first_sentence_text_csv, index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Saved chunk-based CSV to: {output_csv}")
    print(f"Saved first-sentence numeric CSV to: {first_sentence_numeric_csv}")

    if first_sentence_text_csv is not None:
        print(f"Saved first-sentence text CSV to: {first_sentence_text_csv}")

    return chunk_df, first_sentence_numeric_df


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract English content-suitability, first-person narration, "
            "and first-sentence features."
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
        help="Path where the chunk-based output CSV should be saved.",
    )

    parser.add_argument(
        "--first_sentence_numeric_csv",
        required=True,
        help="Path where the first-sentence numeric CSV should be saved.",
    )

    parser.add_argument(
        "--first_sentence_text_csv",
        default=None,
        help="Optional path where first-sentence text CSV should be saved.",
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
        first_sentence_numeric_csv=args.first_sentence_numeric_csv,
        first_sentence_text_csv=args.first_sentence_text_csv,
        file_pattern=args.file_pattern,
    )


if __name__ == "__main__":
    main()