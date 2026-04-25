# =========================
# ENGLISH TEXT CLEANING PIPELINE
# SELECT FILES BY LEADING FILENAME NUMBER
# =========================
#
# PURPOSE
# -------
# This script cleans English plain-text files (for example OCR'd books or
# public-domain texts) and processes only those files whose names begin with a
# numeric identifier.
#
# Example expected filenames:
#   1_book_title.txt
#   27_some_novel.txt
#   100_another_text.txt
#
# The script:
#   1. reads all .txt files from a user-defined input directory
#   2. keeps only files whose leading filename number falls in a chosen range
#   3. applies a series of cleaning rules
#   4. saves cleaned versions to a user-defined output directory
#   5. optionally creates a ZIP archive of the cleaned files
#
# IMPORTANT
# ---------
# Before running the script, edit the configuration section below:
#   - INPUT_DIR
#   - OUTPUT_DIR
#   - ZIP_PATH
#   - START_NO / END_NO
#
# This version avoids platform-specific paths (e.g. Kaggle) so that it can be
# reused by other people locally or in other environments.
#
# =========================

import os
import re
import io
import zipfile
from statistics import median

# --------------------------------------------------
# 1) USER CONFIGURATION
# --------------------------------------------------
# Set the directory containing your input .txt files.
# Example:
# INPUT_DIR = "/Users/your_name/project/raw_books"
INPUT_DIR = "path/to/your/input_txt_files"

# Set the directory where cleaned files should be saved.
# Example:
# OUTPUT_DIR = "/Users/your_name/project/cleaned_books"
OUTPUT_DIR = "path/to/your/output_folder"

# Set the ZIP file path if you want all cleaned files collected into one archive.
# Example:
# ZIP_PATH = "/Users/your_name/project/cleaned_books.zip"
ZIP_PATH = "path/to/your/output_archive.zip"

# Select which files to process based on the number at the START of the filename.
# For example, if START_NO = 1 and END_NO = 100, then files such as
# 1_book.txt, 2_book.txt, ..., 100_book.txt will be included.
START_NO = 1
END_NO = 100

# Number of characters to show in the console preview for each cleaned file.
PREVIEW_CHARS = 600

# Whether to create a ZIP archive after processing.
CREATE_ZIP = True

# --------------------------------------------------
# 2) FILE DISCOVERY
# --------------------------------------------------
def get_book_number(path: str) -> int:
    """
    Extract the leading number from a filename.

    Expected filename format:
        12_title_here.txt

    Returns:
        int: the leading number (e.g. 12)

    Raises:
        ValueError if the filename does not begin with a number followed by "_".
    """
    fname = os.path.basename(path)
    m = re.match(r"(\d+)_", fname)
    if not m:
        raise ValueError(f"Could not extract leading number from filename: {fname}")
    return int(m.group(1))

def collect_txt_files(input_dir: str):
    """
    Collect all .txt files directly inside the input directory.

    Only files ending in .txt are kept.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"INPUT_DIR does not exist or is not a directory: {input_dir}"
        )

    files = []
    for name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, name)
        if os.path.isfile(full_path) and name.lower().endswith(".txt"):
            files.append(full_path)
    return files

all_txt_files = collect_txt_files(INPUT_DIR)

# Keep only files whose names begin with a number in the chosen range.
selected_paths = []
skipped_non_numbered = []

for path in all_txt_files:
    try:
        num = get_book_number(path)
        if START_NO <= num <= END_NO:
            selected_paths.append(path)
    except ValueError:
        skipped_non_numbered.append(os.path.basename(path))

selected_paths = sorted(selected_paths, key=get_book_number)

assert selected_paths, (
    "No matching files were found. Make sure your input directory contains .txt "
    "files named like '1_title.txt', '2_title.txt', etc., and that the chosen "
    "START_NO / END_NO range is correct."
)

# --------------------------------------------------
# 3) OUTPUT SETUP
# --------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 4) HELPERS
# --------------------------------------------------
TERMINAL_PUNCT_RE = re.compile(r'[.!?]["”’\')\]]*$')
NONFINAL_PUNCT_RE = re.compile(r'(?:,|;|:|—|–|-|--|…|\.\.\.)["”’\')\]]*$')

def _strip_trailing_spaces(s: str) -> str:
    return re.sub(r"[ \t]+$", "", s)

def _is_empty_line(s: str) -> bool:
    return re.fullmatch(r"[ \t]*", s) is not None

def _line_words(s: str):
    return re.findall(r"[A-Za-z0-9'’-]+", s)

def _letters_only(s: str) -> str:
    return "".join(re.findall(r"[A-Za-z]", s))

def _is_all_caps(s: str) -> bool:
    letters = _letters_only(s)
    return bool(letters) and letters == letters.upper()

def _all_caps_up_to_n_words(s: str, n: int) -> bool:
    words = _line_words(s)
    return 1 <= len(words) <= n and _is_all_caps(s)

def _title_like_no_punct_end(s: str, max_words: int = 6) -> bool:
    words = _line_words(s)
    if not (1 <= len(words) <= max_words):
        return False
    starts_cap = bool(re.match(r"[A-Z]", s.strip())) or _is_all_caps(s)
    if not starts_cap:
        return False
    return not bool(TERMINAL_PUNCT_RE.search(s))

def ends_with_terminal_punct(s: str) -> bool:
    return bool(TERMINAL_PUNCT_RE.search(s.strip()))

def ends_with_nonfinal_punct(s: str) -> bool:
    return bool(NONFINAL_PUNCT_RE.search(s.strip()))

def starts_with_dialogue_marker(s: str) -> bool:
    t = s.lstrip()
    return bool(re.match(r'^(?:["“”\'‘’]|--|—|–|-)', t))

def is_number_only_line(s: str) -> bool:
    return bool(re.fullmatch(r"\s*\d+\s*", s))

def is_dash_only_line(s: str) -> bool:
    return bool(re.fullmatch(r"\s*[-–—]+\s*", s))

def begins_with_capital(s: str) -> bool:
    return bool(re.match(r"\s*[A-Z]", s))

# --------------------------------------------------
# 5) CLEANING RULES
# --------------------------------------------------
CHAP_KEYS = [
    "CHAPTER", "CHAP", "BOOK", "PART", "VOLUME",
    "PROLOGUE", "EPILOGUE", "PREFACE"
]

def remove_website_tokens(text: str) -> str:
    text = re.sub(r"\b\S+\.(?:com|org|net)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhttps?://\S+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\S*\.edu\S*\b", "", text, flags=re.IGNORECASE)
    return text

def remove_all_underscores(text: str) -> str:
    return text.replace("_", "")

def remove_parenthetical_number_of_number(text: str) -> str:
    return re.sub(r"\(\s*\d+\s+of\s+\d+\s*\)", "", text, flags=re.IGNORECASE)

def remove_parenthetical_note(text: str) -> str:
    return re.sub(r"\(\s*note:.*?\)", "", text, flags=re.IGNORECASE | re.DOTALL)

def normalize_initial_all_caps_lines(lines, max_lines=8, max_words=5):
    out = lines[:]
    checked = 0
    for i, ln in enumerate(out):
        s = ln.strip()
        if not s:
            continue
        checked += 1
        if checked > max_lines:
            break
        words = _line_words(s)
        if 1 <= len(words) <= max_words and _is_all_caps(s):
            lowered = s.lower()
            lowered = lowered[:1].upper() + lowered[1:] if lowered else lowered
            out[i] = lowered
    return out

def remove_beginning_all_caps_punct_lines(lines, max_lines=10, max_words=5):
    out = lines[:]
    checked = 0
    i = 0
    while i < len(out) and checked < max_lines:
        s = out[i].strip()
        if not s:
            i += 1
            continue
        checked += 1
        words = _line_words(s)
        if 1 <= len(words) <= max_words and _is_all_caps(s) and ends_with_terminal_punct(s):
            del out[i]
            continue
        i += 1
    return out

def remove_beginning_copyright_lines(lines, max_lines=10):
    out = lines[:]
    checked = 0
    i = 0
    while i < len(out) and checked < max_lines:
        s = out[i].strip()
        if not s:
            i += 1
            continue
        checked += 1
        if s.startswith("©"):
            del out[i]
            continue
        i += 1
    return out

def remove_beginning_epigraph_block(lines, max_lines=10):
    out = lines[:]
    nonempty = []
    for idx, ln in enumerate(out):
        s = ln.strip()
        if not s:
            continue
        nonempty.append((idx, s))
        if len(nonempty) >= max_lines:
            break

    author_line_idx = None
    for idx, s in nonempty:
        if re.fullmatch(r"(?:[-—]\s*)?(?:[A-Z][^\s]*)(?:\s+[A-Z][^\s]*){0,2}", s) and not ends_with_terminal_punct(s):
            author_line_idx = idx
            break

    if author_line_idx is not None:
        out = out[author_line_idx + 1:]
    return out

def normalize_first_surviving_caps_line(lines, max_words=5):
    out = lines[:]
    for i, ln in enumerate(out):
        s = ln.strip()
        if not s:
            continue
        words = _line_words(s)
        if 1 <= len(words) <= max_words and _is_all_caps(s):
            lowered = s.lower()
            lowered = lowered[:1].upper() + lowered[1:] if lowered else lowered
            out[i] = lowered
        break
    return out

def normalize_caps_prefix_on_first_surviving_line(lines, max_prefix_words=5):
    out = lines[:]
    for i, ln in enumerate(out):
        s = ln.strip()
        if not s:
            continue

        parts = s.split()
        prefix = []
        rest_index = 0

        for j, tok in enumerate(parts):
            letters = re.sub(r"[^A-Za-z]", "", tok)
            if letters and letters.upper() == letters and len(prefix) < max_prefix_words:
                prefix.append(tok)
                rest_index = j + 1
            else:
                break

        if prefix:
            lowered_prefix = " ".join(t.lower() for t in prefix)
            lowered_prefix = lowered_prefix[:1].upper() + lowered_prefix[1:]
            rest = " ".join(parts[rest_index:])
            out[i] = lowered_prefix + (" " + rest if rest else "")
        break
    return out

def remove_first_line_leading_number_dash_if_followed_by_capital(lines):
    out = lines[:]
    for i, ln in enumerate(out):
        s = ln.strip()
        if not s:
            continue
        new_s = re.sub(r"^\d+(?:[-—–]|\s+)(?=[A-Z])", "", s)
        out[i] = new_s
        break
    return out

def remove_first_surviving_parenthetical_line(lines):
    out = lines[:]
    for i, ln in enumerate(out):
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"\([^()]*\)", s):
            del out[i]
        break
    return out

def remove_dot_only_lines(lines):
    pat = re.compile(r"^\s*(?:[.\u2026]\s*){2,}$")
    return [ln for ln in lines if not pat.match(ln)]

def remove_dash_number_dash(lines):
    return [ln for ln in lines if not re.fullmatch(r"\s*-\d{1,6}-\s*", ln)]

def remove_number_only_lines(lines):
    return [ln for ln in lines if not is_number_only_line(ln)]

def remove_dash_only_lines(lines):
    return [ln for ln in lines if not is_dash_only_line(ln)]

def remove_short_number_heading_lines(lines):
    out = []
    for ln in lines:
        s = ln.strip()

        if re.match(r"^\d+\.?\s+", s):
            words = _line_words(s)
            if len(words) > 5 and re.search(r"[.,;:!?—–-]", s):
                s2 = re.sub(r"^\d+\.?\s+", "", s)
                out.append(s2)
                continue

        if re.fullmatch(r"\d+\.?(?:\s+\S+){1,3}", s) and not ends_with_terminal_punct(s):
            continue

        out.append(ln)
    return out

def remove_scanner_lines(lines):
    patterns = [
        r"^\s*Scanned with CS CamScanner\s*$",
        r"^\s*CS CamScanner Scanned with\s*$",
        r"^\s*CS\s*$",
        r"^\s*Scanned with CamScanner\s*$",
        r"^\s*Scanned with\s*$",
        r"^\s*CS CamScanner\s*$",
    ]
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]
    return [ln for ln in lines if not any(r.match(ln) for r in regs)]

def remove_chapter_title_lines(lines):
    out = []
    pat = re.compile(
        r"^\s*chapter\s+\d+(?:\s*[\.:])?(?:\s+\S+){0,10}\s*$",
        flags=re.IGNORECASE
    )
    for ln in lines:
        if pat.match(ln.strip()):
            continue
        out.append(ln)
    return out

def remove_lines_with_keywords(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        tokens = _line_words(s)
        upper = s.upper()
        drop = False
        for kw in CHAP_KEYS:
            if kw in upper:
                idxs = [i for i, t in enumerate(tokens) if t.upper() == kw]
                if idxs:
                    idx = idxs[0]
                    if (len(tokens) - idx - 1) <= 3:
                        drop = True
                        break
        if not drop:
            out.append(ln)
    return out

def remove_starting_labels_english(lines):
    heads = {
        "chapter", "book", "volume", "part", "letter",
        "prologue", "epilogue", "preface", "stave"
    }
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        words = _line_words(s)
        first = words[0].lower() if words else ""
        if first in heads:
            if (not ends_with_terminal_punct(s)) or (len(words) <= 4):
                continue
        out.append(ln)
    return out

def remove_numbered_semicolon_colon_dash_lines(lines):
    pat_start_num = re.compile(r"^\s*\d+\.?\s*")
    out = []
    for ln in lines:
        s = ln.strip()
        if pat_start_num.match(s) and any(ch in s for ch in [";", ":", "-"]):
            continue
        out.append(ln)
    return out

def remove_colon_lists(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        if s.count(":") >= 2 and "." not in s and not ends_with_terminal_punct(s):
            continue
        out.append(ln)
    return out

def remove_glossary_styles(lines):
    out = []
    pat_paren_num = re.compile(r"^\s*\(\d+\)\s+.+[^\.\!\?…]$")
    pat_num_dot = re.compile(r"^\s*\d+\.\s+.+[^\.\!\?…]$")
    pat_num_dot_colon_short = re.compile(r"^\s*\d+\.\s+[^:]{0,40}:")
    pat_roman_dot = re.compile(r"^\s*[IVXLCM]+\.\s+.+[^\.\!\?…]$", re.IGNORECASE)
    pat_bullet_colon = re.compile(r"^\s*[•·]\s*.+:\s*.+[^\.\!\?…]$")
    pat_star_colon = re.compile(r"^\s*\*\s*.+:\s*.+")
    pat_one_char_colon_line = re.compile(r"^\s*.:\s*$")

    for ln in lines:
        s = ln.strip()
        if pat_paren_num.match(s):
            continue
        if pat_num_dot.match(s):
            continue
        if pat_num_dot_colon_short.match(s):
            continue
        if pat_roman_dot.match(s):
            continue
        if pat_bullet_colon.match(s):
            continue
        if pat_star_colon.match(s):
            continue
        if pat_one_char_colon_line.match(s):
            continue
        out.append(ln)
    return out

def remove_small_dot_extension_lines(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        if re.match(r"^\s*[a-z][^\s]*\.[a-z]{1,4}(?:\s|$)", s):
            continue
        out.append(ln)
    return out

def remove_short_pipe_or_short_date_lines(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        words = _line_words(s)
        if s.endswith("|") and len(words) < 5:
            continue
        if re.search(r",\s*\d+\.\s*$", s) and len(words) < 8:
            continue
        out.append(ln)
    return out

def remove_prelude_line(lines):
    return [ln for ln in lines if not re.fullmatch(r"\s*Prelude[.:;!?]?\s*", ln, flags=re.IGNORECASE)]

def remove_introduction_line(lines):
    return [ln for ln in lines if not re.fullmatch(r"\s*Introduction[.:;!?]?\s*", ln, flags=re.IGNORECASE)]

def remove_symbol_heavy_lines(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        if begins_with_capital(s):
            out.append(ln)
            continue
        if ends_with_terminal_punct(s):
            out.append(ln)
            continue

        total_chars = len(s)
        letter_chars = len(re.findall(r"[A-Za-z]", s))
        non_letter_ratio = (total_chars - letter_chars) / max(1, total_chars)

        if non_letter_ratio > 0.5:
            continue
        out.append(ln)
    return out

def remove_contextual_number_title_lines(lines):
    out = []
    for i, ln in enumerate(lines):
        s = ln.strip()

        if re.match(r"^\d+\s+\S+", s):
            prev_line = lines[i - 1].strip() if i > 0 else ""
            next_line = lines[i + 1].strip() if i < len(lines) - 1 else ""

            if prev_line and next_line:
                if (not ends_with_terminal_punct(prev_line)) and (not begins_with_capital(next_line)):
                    continue

        out.append(ln)
    return out

def remove_all_caps_lines_no_punct(lines, max_words=12):
    out = []
    for ln in lines:
        s = _strip_trailing_spaces(ln)
        words = _line_words(s)
        if 1 <= len(words) <= max_words and _is_all_caps(s) and not ends_with_terminal_punct(s):
            continue
        out.append(ln)
    return out

def remove_leading_short_lines(lines, max_head=5, max_words=3):
    i = 0
    while i < min(len(lines), max_head):
        s = lines[i].strip()
        if s and not ends_with_terminal_punct(s) and len(_line_words(s)) <= max_words:
            i += 1
        else:
            break
    return lines[i:]

def normalize_split_caps_at_line_start(text: str) -> str:
    def fix_line(ln: str) -> str:
        for _ in range(3):
            ln = re.sub(r"^([A-Z])\s+([A-Z])\b", r"\1\2", ln)
        return ln
    return "\n".join(fix_line(ln) for ln in text.splitlines())

def remove_symbols_late(text: str) -> str:
    text = text.replace("·", "").replace("▪", "").replace("※", "").replace("♦", "").replace("•", "")
    text = text.replace("*", "")
    text = text.replace("/", "")
    text = re.sub(r"[–—]{2,}", "—", text)
    return text

def fix_hyphenated_linebreaks(text: str) -> str:
    return re.sub(r"([A-Za-z])-\n\s*([A-Za-z])", r"\1\2", text)

def remove_trailing_digits_in_words(text: str) -> str:
    return re.sub(r"(?<=[A-Za-z])\d+\b", "", text)

def remove_digits_attached_to_letters_anywhere(text: str) -> str:
    # Keep ordinals like 5th, 21st, 2nd
    text = re.sub(r"(?<=[A-Za-z])\d+(?!\b)", "", text)
    text = re.sub(r"\d+(?=(?!th\b|st\b|nd\b)[A-Za-z])", "", text)
    return text

def remove_inline_small_numbers_eol_after_punct(text: str) -> str:
    return re.sub(r"(?<=[.!?…])\s*\b\d{1,2}\b\s*$", "", text, flags=re.MULTILINE)

def safe_remove_square_brackets(text: str) -> str:
    text = re.sub(r"\[[^\]\n]*\]", "", text)
    text = re.sub(r"\[[^\]]*\n[^\]]*\]", "", text)
    return text

def remove_title_like(lines):
    out = []
    for i, ln in enumerate(lines):
        s = _strip_trailing_spaces(ln)
        prev_empty = (i == 0) or _is_empty_line(_strip_trailing_spaces(lines[i - 1]))
        next_empty = (i == len(lines) - 1) or _is_empty_line(_strip_trailing_spaces(lines[i + 1]))
        drop = False

        if _title_like_no_punct_end(s, max_words=6):
            drop = True

        if not drop and prev_empty and next_empty and _all_caps_up_to_n_words(s, 6):
            drop = True

        if not drop:
            out.append(ln)
    return out

def remove_tiny_ocr_crumbs(lines):
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        if len(s) <= 2 and not re.search(r"[A-Za-z0-9]", s):
            continue
        out.append(ln)
    return out

# --------------------------------------------------
# 6) BLANK-LINE HEALING + RE-FLOW
# --------------------------------------------------
def collapse_spurious_blank_lines(text: str) -> str:
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if i > 0 and i < len(lines) - 1 and _is_empty_line(lines[i]):
            prev_line = out[-1].strip() if out else ""
            next_line = lines[i + 1].strip()

            if prev_line and next_line:
                if (not ends_with_terminal_punct(prev_line)) or ends_with_nonfinal_punct(prev_line):
                    i += 1
                    continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)

def should_merge_lines(curr: str, nxt: str, median_len: float) -> bool:
    c = curr.strip()
    n = nxt.strip()

    if not c or not n:
        return False

    if re.search(r'["”]\s*$', c) and re.match(r"^[a-z]", n):
        return True

    if re.match(r"^--?[a-z]", n) and not ends_with_terminal_punct(c):
        return True

    if starts_with_dialogue_marker(n):
        return False

    if not ends_with_terminal_punct(c):
        return True

    if ends_with_nonfinal_punct(c):
        return True

    curr_len = len(c)
    if median_len >= 35 and curr_len >= max(35, 0.72 * median_len):
        return True

    return False

def reflow_wrapped_lines(text: str) -> str:
    lines = text.splitlines()
    out = []
    i = 0

    while i < len(lines):
        if _is_empty_line(lines[i]):
            i += 1
            continue

        block = []
        while i < len(lines) and not _is_empty_line(lines[i]):
            block.append(_strip_trailing_spaces(lines[i]))
            i += 1

        lengths = [len(ln.strip()) for ln in block if ln.strip()]
        med = median(lengths) if lengths else 0

        if not block:
            continue

        current = block[0].strip()

        for nxt in block[1:]:
            nxt_stripped = nxt.strip()

            if should_merge_lines(current, nxt_stripped, med):
                if current.endswith("-"):
                    current = current[:-1] + nxt_stripped
                else:
                    current = current + " " + nxt_stripped
            else:
                out.append(current)
                current = nxt_stripped

        out.append(current)

    return "\n".join(out)

def final_spacing_and_blanklines(text: str) -> str:
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r" {2,}", " ", text)
    text = "\n".join(_strip_trailing_spaces(ln) for ln in text.splitlines())
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines).strip()

# --------------------------------------------------
# 7) MAIN CLEANER
# --------------------------------------------------
def clean_text_english(raw: str, filename: str) -> str:
    """
    Apply the full cleaning pipeline to one text.
    """
    txt = raw.replace("\r\n", "\n").replace("\r", "\n")

    txt = remove_website_tokens(txt)
    txt = remove_all_underscores(txt)
    txt = remove_parenthetical_number_of_number(txt)
    txt = remove_parenthetical_note(txt)

    lines = txt.splitlines()

    lines = remove_leading_short_lines(lines, max_head=5, max_words=3)
    lines = normalize_initial_all_caps_lines(lines, max_lines=8, max_words=5)
    lines = remove_beginning_all_caps_punct_lines(lines, max_lines=10, max_words=5)
    lines = remove_beginning_copyright_lines(lines, max_lines=10)
    lines = remove_beginning_epigraph_block(lines, max_lines=10)

    lines = remove_number_only_lines(lines)
    lines = remove_dash_only_lines(lines)
    lines = remove_short_number_heading_lines(lines)
    lines = remove_chapter_title_lines(lines)
    lines = remove_lines_with_keywords(lines)
    lines = remove_starting_labels_english(lines)
    lines = remove_dash_number_dash(lines)
    lines = remove_scanner_lines(lines)
    lines = remove_glossary_styles(lines)
    lines = remove_colon_lists(lines)
    lines = remove_dot_only_lines(lines)
    lines = remove_numbered_semicolon_colon_dash_lines(lines)
    lines = remove_small_dot_extension_lines(lines)
    lines = remove_short_pipe_or_short_date_lines(lines)
    lines = remove_prelude_line(lines)
    lines = remove_introduction_line(lines)
    lines = remove_contextual_number_title_lines(lines)
    lines = remove_symbol_heavy_lines(lines)
    lines = remove_all_caps_lines_no_punct(lines, max_words=12)
    lines = remove_title_like(lines)
    lines = remove_tiny_ocr_crumbs(lines)

    lines = remove_first_surviving_parenthetical_line(lines)
    lines = normalize_first_surviving_caps_line(lines, max_words=5)
    lines = normalize_caps_prefix_on_first_surviving_line(lines, max_prefix_words=5)
    lines = remove_first_line_leading_number_dash_if_followed_by_capital(lines)

    txt = "\n".join(lines)

    txt = fix_hyphenated_linebreaks(txt)
    txt = remove_trailing_digits_in_words(txt)
    txt = remove_digits_attached_to_letters_anywhere(txt)
    txt = re.sub(r"\(\d+\)", "", txt)
    txt = safe_remove_square_brackets(txt)
    txt = remove_inline_small_numbers_eol_after_punct(txt)
    txt = remove_symbols_late(txt)

    # Special case preserved from original script:
    # if a filename starts with "67_", remove long standalone numbers.
    # Remove or generalize this if it is too corpus-specific for your project.
    if filename.startswith("67_"):
        txt = re.sub(r"\b\d{3,}\b", "", txt)

    txt = normalize_split_caps_at_line_start(txt)
    txt = collapse_spurious_blank_lines(txt)
    txt = reflow_wrapped_lines(txt)
    txt = final_spacing_and_blanklines(txt)

    return txt

# --------------------------------------------------
# 8) PROCESS SELECTED FILES
# --------------------------------------------------
written = []
total_input_chars = 0
total_output_chars = 0

print(f"[INFO] Input directory:  {INPUT_DIR}")
print(f"[INFO] Output directory: {OUTPUT_DIR}")
print(f"[INFO] Processing files numbered {START_NO}-{END_NO}.\n")

if skipped_non_numbered:
    print("[INFO] Skipped .txt files without a leading numeric prefix:")
    for name in sorted(skipped_non_numbered):
        print(" -", name)
    print()

print("[INFO] Selected files in order:")
for p in selected_paths:
    print(" -", os.path.basename(p))
print()

for idx, path in enumerate(selected_paths, start=1):
    fname = os.path.basename(path)
    stem, ext = os.path.splitext(fname)
    out_fname = f"{stem}_cleaned{ext}"
    out_path = os.path.join(OUTPUT_DIR, out_fname)

    print("=" * 110)
    print(f"[{idx}/{len(selected_paths)}] Processing: {fname}")

    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    cleaned = clean_text_english(raw, fname)

    with io.open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    total_input_chars += len(raw)
    total_output_chars += len(cleaned)
    reduction = 100 * (1 - (len(cleaned) / max(1, len(raw))))

    preview = cleaned[:PREVIEW_CHARS].replace("\n", " ")
    if len(cleaned) > PREVIEW_CHARS:
        preview += "..."

    print(f"Saved to:      {out_path}")
    print(f"Input chars:   {len(raw):,}")
    print(f"Output chars:  {len(cleaned):,}")
    print(f"Reduction:     {reduction:.2f}%")
    print("Preview:")
    print(preview)
    print()

    written.append((out_fname, out_path))

# --------------------------------------------------
# 9) OPTIONAL ZIP ARCHIVE
# --------------------------------------------------
if CREATE_ZIP:
    zip_dir = os.path.dirname(ZIP_PATH)
    if zip_dir:
        os.makedirs(zip_dir, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for out_fname, out_path in written:
            zf.write(out_path, arcname=out_fname)

    print("=" * 110)
    print("[DONE]")
    print(f"Processed files:     {len(written)}")
    print(f"Total input chars:   {total_input_chars:,}")
    print(f"Total output chars:  {total_output_chars:,}")
    print(f"Output folder:       {OUTPUT_DIR}")
    print(f"ZIP archive:         {ZIP_PATH}")
else:
    print("=" * 110)
    print("[DONE]")
    print(f"Processed files:     {len(written)}")
    print(f"Total input chars:   {total_input_chars:,}")
    print(f"Total output chars:  {total_output_chars:,}")
    print(f"Output folder:       {OUTPUT_DIR}")
    print("ZIP archive:         not created (CREATE_ZIP = False)")