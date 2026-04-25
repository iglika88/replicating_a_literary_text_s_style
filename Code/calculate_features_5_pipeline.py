"""
English lexical-specificity, lexical-class, and modality features.

Features:
- ttr_word
- ttr_lemma
- conjuncts_per100
- downtoners_per100
- hedges_per100
- amplifiers_per100
- emphatics_per100
- discourse_particles_per100
- demonstrative_dets_per100
- demonstrative_prons_per100
- possibility_per100
- necessity_per100
- predictive_per100

Note:
- Mean word length is intentionally excluded here.

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
# Lexical-class inventories
# ============================================================

def compile_mwes(pairs):
    return [(re.compile(pattern, re.IGNORECASE), label) for pattern, label in pairs]


LEX_EN = {
    "conjuncts_lemmas": {
        "however", "nevertheless", "nonetheless", "therefore", "thus",
        "furthermore", "moreover", "instead", "otherwise", "besides",
        "hence", "consequently", "meanwhile", "afterward", "afterwards",
    },
    "conjuncts_mwe": compile_mwes([
        (r"\bas a result\b", "conjuncts_per100"),
        (r"\bon the other hand\b", "conjuncts_per100"),
        (r"\bin contrast\b", "conjuncts_per100"),
    ]),

    "downtoners_lemmas": {
        "slightly", "somewhat", "barely", "hardly", "mildly",
        "moderately", "rather",
    },
    "downtoners_mwe": compile_mwes([
        (r"\ba little\b", "downtoners_per100"),
        (r"\ba bit\b", "downtoners_per100"),
        (r"\bkinda\b", "downtoners_per100"),
    ]),

    "hedges_lemmas": {
        "about", "around", "approximately", "almost", "nearly",
        "roughly", "generally", "virtually",
    },
    "hedges_mwe": compile_mwes([
        (r"\bmore or less\b", "hedges_per100"),
        (r"\bkind of\b", "hedges_per100"),
        (r"\bsort of\b", "hedges_per100"),
    ]),

    "amplifiers_lemmas": {
        "very", "really", "so", "truly", "absolutely", "extremely",
        "highly", "too", "super", "totally",
    },
    "amplifiers_mwe": compile_mwes([
        (r"\bby far\b", "amplifiers_per100"),
    ]),

    "emphatics_lemmas": {
        "indeed", "certainly", "clearly", "obviously", "frankly",
        "definitely", "surely",
    },
    "emphatics_mwe": compile_mwes([
        (r"\bfor sure\b", "emphatics_per100"),
    ]),

    "discourse_start_mwe": compile_mwes([
        (r"^\s*well\b", "discourse_particles_per100"),
        (r"^\s*so\b", "discourse_particles_per100"),
        (r"^\s*anyway\b", "discourse_particles_per100"),
        (r"^\s*now\b", "discourse_particles_per100"),
    ]),

    "discourse_start_single": {"well", "so", "anyway", "now"},

    "demonstrative_det": {"this", "that", "these", "those"},
    "demonstrative_pron": {"this", "that", "these", "those"},
}


# ============================================================
# Modality inventories
# ============================================================

POSSIBILITY_LEMMAS = {"can", "could", "may", "might"}
NECESSITY_CORE_LEMMAS = {"must", "should", "ought"}
NECESSITY_EXTRA_LEMMAS = {"have", "need"}
PREDICTIVE_LEMMAS = {"will", "shall", "would"}

RE_GOING_TO = re.compile(
    r"\b(am|is|are|was|were|be|been|being)\s+going\s+to\s+\w+",
    re.IGNORECASE,
)


FEATURE_NAMES = [
    "ttr_word",
    "ttr_lemma",
    "conjuncts_per100",
    "downtoners_per100",
    "hedges_per100",
    "amplifiers_per100",
    "emphatics_per100",
    "discourse_particles_per100",
    "demonstrative_dets_per100",
    "demonstrative_prons_per100",
    "possibility_per100",
    "necessity_per100",
    "predictive_per100",
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
    return not tok.is_space and not tok.is_punct


def is_wordlike_token(tok):
    """
    Wordlike policy for lexical specificity:
    alphabetic tokens, excluding spaces, punctuation, symbols, and pure numbers.
    """
    if tok.is_space or tok.is_punct:
        return False
    if tok.like_num or tok.pos_ == "SYM":
        return False
    return any(ch.isalpha() for ch in tok.text)


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


def split_into_chunks(items, chunk_size=CHUNK_SIZE):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# ============================================================
# Lexical specificity
# ============================================================

def compute_ttr_features(chunk_tokens):
    """Compute surface and lemma type-token ratio for one 100-word chunk."""
    if not chunk_tokens:
        return {
            "ttr_word": 0.0,
            "ttr_lemma": 0.0,
        }

    surfaces = [tok.text.lower() for tok in chunk_tokens]
    lemmas = [lemma(tok) for tok in chunk_tokens]

    return {
        "ttr_word": (len(set(surfaces)) / len(surfaces)) * 100.0,
        "ttr_lemma": (len(set(lemmas)) / len(lemmas)) * 100.0,
    }


# ============================================================
# Lexical-class features
# ============================================================

def is_sentence_initial_lexical_token(tok):
    """Check whether token is the first non-space, non-punctuation token in its sentence."""
    for candidate in tok.sent:
        if candidate.is_space or candidate.is_punct:
            continue
        return candidate.i == tok.i
    return False


def is_conjunct(tok):
    return lemma(tok) in LEX_EN["conjuncts_lemmas"] and tok.pos_ in {"ADV", "SCONJ", "CCONJ"}


def is_downtoner(tok):
    tok_lemma = lemma(tok)

    if tok_lemma not in LEX_EN["downtoners_lemmas"]:
        return False

    if tok.pos_ == "ADV":
        return True

    if tok_lemma == "rather" and tok.dep_ in {"advmod", "mod"}:
        return True

    return False


def is_hedge(tok):
    tok_lemma = lemma(tok)

    if tok_lemma not in LEX_EN["hedges_lemmas"]:
        return False

    if tok.pos_ == "ADV":
        return True

    if tok_lemma in {"about", "around"} and tok.pos_ in {"ADV", "ADP"}:
        return True

    return False


def is_amplifier(tok):
    tok_lemma = lemma(tok)

    if tok_lemma not in LEX_EN["amplifiers_lemmas"]:
        return False

    if tok.pos_ != "ADV":
        return False

    if tok_lemma == "so" and is_sentence_initial_lexical_token(tok):
        return False

    if tok.dep_ == "advmod":
        return True

    if tok.head is not None and tok.head.pos_ in {"ADJ", "ADV", "VERB"}:
        return True

    return False


def is_emphatic(tok):
    return lemma(tok) in LEX_EN["emphatics_lemmas"] and tok.pos_ == "ADV"


def is_discourse_particle_sentence(sentence):
    """Detect discourse starters such as 'Well, ...', 'So, ...', 'Anyway, ...'."""
    sentence_text = sentence.text.strip().lower()

    for pattern, _ in LEX_EN["discourse_start_mwe"]:
        if pattern.search(sentence_text):
            return True

    for tok in sentence:
        if tok.is_space or tok.is_punct:
            continue
        return lemma(tok) in LEX_EN["discourse_start_single"]

    return False


def classify_lexical_classes_in_sentence(sentence):
    """Return lexical-class counts for one sentence."""
    counts = {
        "conjuncts_per100": 0,
        "downtoners_per100": 0,
        "hedges_per100": 0,
        "amplifiers_per100": 0,
        "emphatics_per100": 0,
        "discourse_particles_per100": 0,
        "demonstrative_dets_per100": 0,
        "demonstrative_prons_per100": 0,
    }

    sentence_text = sentence.text.strip().lower()

    for tok in sentence:
        tok_lemma = lemma(tok)

        if is_conjunct(tok):
            counts["conjuncts_per100"] += 1

        if is_downtoner(tok):
            counts["downtoners_per100"] += 1

        if is_hedge(tok):
            counts["hedges_per100"] += 1

        if is_amplifier(tok):
            counts["amplifiers_per100"] += 1

        if is_emphatic(tok):
            counts["emphatics_per100"] += 1

        if tok.pos_ == "DET" and tok_lemma in LEX_EN["demonstrative_det"]:
            counts["demonstrative_dets_per100"] += 1

        if tok.pos_ == "PRON" and tok_lemma in LEX_EN["demonstrative_pron"]:
            counts["demonstrative_prons_per100"] += 1

    for pattern, feature_name in LEX_EN["conjuncts_mwe"]:
        if pattern.search(sentence_text):
            counts[feature_name] += 1
            break

    for pattern, feature_name in LEX_EN["downtoners_mwe"]:
        if pattern.search(sentence_text):
            counts[feature_name] += 1
            break

    for pattern, feature_name in LEX_EN["hedges_mwe"]:
        if pattern.search(sentence_text):
            counts[feature_name] += 1
            break

    for pattern, feature_name in LEX_EN["amplifiers_mwe"]:
        if pattern.search(sentence_text):
            counts[feature_name] += 1
            break

    for pattern, feature_name in LEX_EN["emphatics_mwe"]:
        if pattern.search(sentence_text):
            counts[feature_name] += 1
            break

    if is_discourse_particle_sentence(sentence):
        counts["discourse_particles_per100"] += 1

    return counts


# ============================================================
# Modality features
# ============================================================

def has_infinitive_to_right(head, window=6):
    """Check whether a modal-like token has an infinitive complement nearby."""
    for child in head.children:
        if child.pos_ in {"VERB", "AUX"} and morph_has(child, "VerbForm", "Inf"):
            return True

    sentence_tokens = list(head.sent)
    start_idx = head.i - sentence_tokens[0].i

    for j in range(start_idx + 1, min(start_idx + 1 + window, len(sentence_tokens))):
        tok = sentence_tokens[j]
        if tok.pos_ in {"VERB", "AUX"} and morph_has(tok, "VerbForm", "Inf"):
            return True

    return False


def sentence_has_to_infinitive(sentence):
    """Detect to + infinitive in a sentence."""
    sentence_tokens = list(sentence)

    for i in range(len(sentence_tokens) - 1):
        first = sentence_tokens[i]
        second = sentence_tokens[i + 1]

        if (
            first.lower_ == "to"
            and second.pos_ in {"VERB", "AUX"}
            and morph_has(second, "VerbForm", "Inf")
        ):
            return True

    return False


def classify_modality_in_sentence(sentence):
    """
    Count broad modality categories.

    Each category is counted at most once per sentence.
    """
    counts = {
        "possibility_per100": 0,
        "necessity_per100": 0,
        "predictive_per100": 0,
    }

    sentence_text = sentence.text.strip().lower()

    possibility_hit = False
    necessity_hit = False
    predictive_hit = False

    for tok in sentence:
        tok_lemma = lemma(tok)
        is_verbal = tok.pos_ in {"AUX", "VERB"}

        if (
            not possibility_hit
            and is_verbal
            and tok_lemma in POSSIBILITY_LEMMAS
            and has_infinitive_to_right(tok)
        ):
            possibility_hit = True

        if (
            not necessity_hit
            and is_verbal
            and tok_lemma in NECESSITY_CORE_LEMMAS
            and has_infinitive_to_right(tok)
        ):
            necessity_hit = True

        if (
            not necessity_hit
            and is_verbal
            and tok_lemma in NECESSITY_EXTRA_LEMMAS
            and sentence_has_to_infinitive(sentence)
            and has_infinitive_to_right(tok)
        ):
            necessity_hit = True

        if (
            not predictive_hit
            and is_verbal
            and tok_lemma in PREDICTIVE_LEMMAS
            and has_infinitive_to_right(tok)
        ):
            predictive_hit = True

    if not predictive_hit and RE_GOING_TO.search(sentence_text):
        predictive_hit = True

    if possibility_hit:
        counts["possibility_per100"] = 1

    if necessity_hit:
        counts["necessity_per100"] = 1

    if predictive_hit:
        counts["predictive_per100"] = 1

    return counts


# ============================================================
# Book processing
# ============================================================

def process_book(text, nlp, chunk_size=CHUNK_SIZE, char_batch_size=CHAR_BATCH_SIZE):
    """Process one literary work and return book-level aggregate features."""
    text_batches = split_text_into_char_batches(text, batch_size=char_batch_size)

    all_wordlike_tokens = []
    global_word_index = 0

    sentence_level_features = [
        "conjuncts_per100",
        "downtoners_per100",
        "hedges_per100",
        "amplifiers_per100",
        "emphatics_per100",
        "discourse_particles_per100",
        "demonstrative_dets_per100",
        "demonstrative_prons_per100",
        "possibility_per100",
        "necessity_per100",
        "predictive_per100",
    ]

    feature_positions = {name: [] for name in sentence_level_features}

    for doc in nlp.pipe(text_batches, batch_size=1):
        for sentence in doc.sents:
            sentence_tokens = list(sentence)
            word_tokens = [tok for tok in sentence_tokens if is_word_token(tok)]
            sentence_word_count = len(word_tokens)

            if sentence_word_count == 0:
                continue

            sentence_start = global_word_index

            lexical_counts = classify_lexical_classes_in_sentence(sentence)
            modality_counts = classify_modality_in_sentence(sentence)

            combined_counts = {**lexical_counts, **modality_counts}

            for feature_name, count in combined_counts.items():
                if count > 0:
                    feature_positions[feature_name].extend([sentence_start] * count)

            for tok in sentence_tokens:
                if is_wordlike_token(tok):
                    all_wordlike_tokens.append(tok)

            global_word_index += sentence_word_count

    token_total = global_word_index
    chunk_count = math.ceil(token_total / chunk_size) if token_total > 0 else 0

    result = {
        "chunk_size_words": chunk_size,
        "chunk_count_used": chunk_count,
        "token_total": token_total,
        "n_tokens_total": len(all_wordlike_tokens),
    }

    # Lexical specificity: calculated over wordlike-token chunks.
    ttr_word_values = []
    ttr_lemma_values = []

    for chunk in split_into_chunks(all_wordlike_tokens, chunk_size=chunk_size):
        stats = compute_ttr_features(chunk)
        ttr_word_values.append(stats["ttr_word"])
        ttr_lemma_values.append(stats["ttr_lemma"])

    result.update(aggregate_feature(ttr_word_values, "ttr_word"))
    result.update(aggregate_feature(ttr_lemma_values, "ttr_lemma"))

    # Lexical-class and modality features: assigned to sentence-start chunks.
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
            if i < chunk_count - 1:
                current_chunk_size = chunk_size
            else:
                current_chunk_size = token_total - (i * chunk_size)
                if current_chunk_size <= 0:
                    current_chunk_size = chunk_size

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
        "n_tokens_total",

        "ttr_word_avg",
        "ttr_word_min",
        "ttr_word_max",
        "ttr_word_sd",

        "ttr_lemma_avg",
        "ttr_lemma_min",
        "ttr_lemma_max",
        "ttr_lemma_sd",
    ]

    for feature in [
        "conjuncts_per100",
        "downtoners_per100",
        "hedges_per100",
        "amplifiers_per100",
        "emphatics_per100",
        "discourse_particles_per100",
        "demonstrative_dets_per100",
        "demonstrative_prons_per100",
        "possibility_per100",
        "necessity_per100",
        "predictive_per100",
    ]:
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
        description="Extract English lexical-specificity, lexical-class, and modality features."
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