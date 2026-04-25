# =========================
# TOP OVERUSED WORDS IN TEXTS
# COMPARED TO A REFERENCE FREQUENCY LIST
# =========================
#
# PURPOSE
# -------
# This script compares one or more English plain-text files against a
# reference frequency list and identifies the words that are used
# disproportionately often in each text.
#
# In other words, it finds words whose relative frequency in a given text
# is higher than would be expected based on the reference list.
#
# The output for each text is a ranked top-N list such as:
#   1. whale   240.5% (used much more often than expected)
#
# HOW IT WORKS
# ------------
# For each word:
#   - compute its relative frequency in the text
#   - compute its relative frequency in the reference list
#   - divide text frequency by reference frequency
#   - keep words whose ratio is greater than 1
#   - rank them by percentage deviation
#
# IMPORTANT
# ---------
# This script assumes that the file:
#
#   frequency_list_10k.txt
#
# is present in the CURRENT WORKING DIRECTORY (that is, in the same folder
# from which you run the script, unless you change the path below).
#
# The frequency list is expected to be tab-separated and to contain columns:
#   Rank    Word    Count (per billion)
#
# EXAMPLE INPUT FILES
# -------------------
# Replace the example filenames below with your own text files.
# These should be cleaned plain-text files encoded in UTF-8.
#
# =========================

import os
import re
import pandas as pd
from collections import Counter, defaultdict

# --------------------------------------------------
# 1) USER CONFIGURATION
# --------------------------------------------------
# Add the text files you want to analyse here.
# Example:
# TEXT_PATHS = [
#     "book1_cleaned.txt",
#     "book2_cleaned.txt",
#     "book3_cleaned.txt",
# ]
TEXT_PATHS = [
    "text1_cleaned.txt",
    "text2_cleaned.txt",
    "text3_cleaned.txt",
    "text4_cleaned.txt",
    "text5_cleaned.txt",
]

# The reference frequency list is assumed to be in the current directory.
# If not, replace this with the full or relative path to the file.
FREQ_PATH = "frequency_list_10k.txt"

# Number of overused words to display per text.
TOP_N = 10

# Exclude words capitalized this proportion of the time or more.
# This helps remove probable proper names.
CAP_THRESHOLD = 0.90

# Ignore words occurring fewer than this many times in a text.
# This reduces noise from rare words.
MIN_COUNT = 5

# --------------------------------------------------
# 2) LOAD REFERENCE FREQUENCY LIST
# --------------------------------------------------
if not os.path.isfile(FREQ_PATH):
    raise FileNotFoundError(
        f"Reference frequency list not found: {FREQ_PATH}\n"
        f"Make sure 'frequency_list_10k.txt' is in the current directory "
        f"or update FREQ_PATH."
    )

df = pd.read_csv(FREQ_PATH, sep="\t")

# Clean column names in case of hidden spaces
df.columns = [c.strip() for c in df.columns]

required_columns = {"Word", "Count (per billion)"}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(
        f"Missing required columns in frequency list: {missing}\n"
        f"Expected columns include: Word, Count (per billion)"
    )

# Convert frequency counts to numeric
df["Count (per billion)"] = (
    df["Count (per billion)"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["Count (per billion)"] = pd.to_numeric(df["Count (per billion)"], errors="coerce")

# Drop incomplete/broken rows
df = df.dropna(subset=["Word", "Count (per billion)"]).copy()

# Normalize words to lowercase
df["Word"] = df["Word"].astype(str).str.strip().str.lower()

# Convert reference frequencies into relative proportions
total_ref = df["Count (per billion)"].sum()
df["ref_pct"] = df["Count (per billion)"] / total_ref

# Dictionary: word -> reference relative frequency
ref_freq = dict(zip(df["Word"], df["ref_pct"]))

# --------------------------------------------------
# 3) TOKENIZER
# --------------------------------------------------
def tokenize_with_case(text):
    """
    Extract word-like tokens while preserving original capitalization.

    Example:
        "Captain Ahab said Hello." -> ["Captain", "Ahab", "said", "Hello"]

    We preserve case because later we estimate whether a word is likely
    to be a proper name by checking how often it appears capitalized.
    """
    return re.findall(r"[A-Za-z']+", text)

# --------------------------------------------------
# 4) ANALYSIS FUNCTION
# --------------------------------------------------
def analyse_text(path, ref_freq, top_n=10, cap_threshold=0.90, min_count=5):
    """
    Analyse one text file and return the most overused words compared
    to the reference frequency list.

    Returns a sorted list of dictionaries with:
        - word
        - deviation_pct
        - book_pct
        - ref_pct
        - count
        - cap_ratio
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Text file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    raw_tokens = tokenize_with_case(text)
    lower_tokens = [tok.lower() for tok in raw_tokens]

    total_words = len(lower_tokens)
    counts = Counter(lower_tokens)

    # Track how often each word appears capitalized
    total_occ = defaultdict(int)
    cap_occ = defaultdict(int)

    for tok in raw_tokens:
        low = tok.lower()
        total_occ[low] += 1
        if tok[0].isupper():
            cap_occ[low] += 1

    results = []

    for word, count in counts.items():
        # Only compare words that exist in the reference list
        if word not in ref_freq:
            continue

        # Ignore very rare words
        if count < min_count:
            continue

        # Exclude likely proper names
        cap_ratio = cap_occ[word] / total_occ[word]
        if cap_ratio >= cap_threshold:
            continue

        book_pct = count / total_words
        ref_pct = ref_freq[word]

        if ref_pct <= 0:
            continue

        ratio = book_pct / ref_pct

        # Keep only words that are overused relative to the reference
        if ratio > 1:
            deviation = (ratio - 1) * 100

            results.append({
                "word": word,
                "deviation_pct": deviation,
                "book_pct": book_pct,
                "ref_pct": ref_pct,
                "count": count,
                "cap_ratio": cap_ratio,
            })

    results = sorted(results, key=lambda x: x["deviation_pct"], reverse=True)
    return results[:top_n], total_words

# --------------------------------------------------
# 5) RUN ANALYSIS
# --------------------------------------------------
for path in TEXT_PATHS:
    print("=" * 100)
    print(f"Processing: {os.path.basename(path)}\n")

    try:
        top_results, total_words = analyse_text(
            path=path,
            ref_freq=ref_freq,
            top_n=TOP_N,
            cap_threshold=CAP_THRESHOLD,
            min_count=MIN_COUNT,
        )

        print(f"Total words: {total_words:,}\n")

        if not top_results:
            print("No matching overused words found.\n")
            continue

        for i, row in enumerate(top_results, 1):
            print(
                f"{i}. {row['word']:<15} "
                f"{row['deviation_pct']:.1f}% "
                f"(text: {row['book_pct']:.4%} vs ref: {row['ref_pct']:.4%}, "
                f"count: {row['count']}, cap_ratio: {row['cap_ratio']:.2f})"
            )

        print()

    except Exception as e:
        print(f"[ERROR] Could not process {path}")
        print(e)
        print()