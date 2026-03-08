 

import re
from typing import List, Tuple

import numpy as np




_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_PATH = re.compile(r"(?:[A-Za-z]:\\|/)(?:[\w.\-]+[/\\])+[\w.\-]*")
_RE_HEADER_LINE = re.compile(r"^[A-Za-z\-]+:\s.*$", re.MULTILINE)
_RE_NONALPHA_LINE = re.compile(r"^[^a-zA-Z]*$", re.MULTILINE)
_RE_PGP = re.compile(
    r"-----BEGIN PGP.*?-----END PGP.*?-----", re.DOTALL
)
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_QUOTE_LINE = re.compile(r"^[>|]+.*$", re.MULTILINE)
_RE_BOILERPLATE = re.compile(
    r"(?:writes?|wrote|said|posted|article|newsgroup|followup|"
    r"in article|distribution|organization|lines):\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def clean_document(text: str) -> str:
    if not text or not text.strip():
        return ""

    text = _RE_PGP.sub("", text)

    text = _RE_QUOTE_LINE.sub("", text)

    text = _RE_EMAIL.sub("", text)

    text = _RE_URL.sub("", text)

    text = _RE_PATH.sub("", text)

    lines = text.split("\n")
    cleaned_lines = []
    header_zone = True
    for i, line in enumerate(lines):
        if header_zone and (i < 8) and _RE_HEADER_LINE.match(line.strip()):
            continue
        else:
            header_zone = False
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    text = _RE_NONALPHA_LINE.sub("", text)

    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    text = _RE_MULTI_SPACE.sub(" ", text)

    return text.strip()


def clean_corpus(
    documents: List[str],
    categories: np.ndarray,
    category_names: List[str],
    min_length: int = 30,
) -> Tuple[List[str], List[int], List[int]]:
    cleaned_docs = []
    valid_indices = []
    valid_categories = []
    discarded_count = 0

    for i, doc in enumerate(documents):
        cleaned = clean_document(doc)

        if len(cleaned) >= min_length:
            cleaned_docs.append(cleaned)
            valid_indices.append(i)
            valid_categories.append(int(categories[i]))
        else:
            discarded_count += 1


    return cleaned_docs, valid_indices, valid_categories
