"""
esp_utils.py
------------
IO helpers, text normalization, and a legal-aware sentence splitter.
"""
import json
import re


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_ws(s: str) -> str:
    """Collapse whitespace. Used for robust substring matching."""
    return re.sub(r"\s+", " ", s).strip()


def normalize_section(s: str) -> str:
    """Force any section reference into the exact 'IPC <num>' gold format.

    Handles inputs like '302', 'Section 302', 'S.302', 'IPC-302', '498-A', '498a'.
    Preserves alphabetic suffixes (498A, 304B) and upper-cases them.
    """
    if not s:
        return s
    s = s.upper()
    m = re.search(r"(\d+\s*[A-Z]?)", s)
    if not m:
        return s.strip()
    num = re.sub(r"\s+", "", m.group(1))          # '498 A' -> '498A'
    return f"IPC {num}"


# Abbreviations that must NOT trigger a sentence break in Indian judgments.
_ABBREV = [
    "pw", "dw", "cw", "ext", "no", "nos", "rs", "sec", "secs", "art", "arts",
    "vs", "v", "ors", "anr", "smt", "sri", "shri", "mr", "mrs", "dr", "m", "s",
    "i.e", "e.g", "etc", "p.m", "a.m", "u.p", "j", "hon'ble",
]
def split_sentences(text: str):
    """Split legal text into sentences, protecting common abbreviations.

    Returns a list of (sentence_text) preserving the original substring as closely
    as possible (whitespace-normalized).
    """
    text = text.strip()
    if not text:
        return []
    # Protect abbreviations by temporarily masking the following period.
    protected = text
    for ab in _ABBREV:
        protected = re.sub(
            r"(?i)\b(%s)\." % re.escape(ab), r"\1<DOT>", protected
        )
    # Split on ., ! or ? followed by whitespace + capital / digit / quote.
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'A-Z0-9(])", protected)
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".")
        p = norm_ws(p)
        if p:
            out.append(p)
    return out
