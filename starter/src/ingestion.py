"""Read and normalize source data."""
import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 20


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page, normalize its text and keep page number and source.

    Pages with no text are detected and recorded; OCR, when needed, is a separate step.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)
    if reader.is_encrypted and reader.decrypt("") == 0:
        raise ValueError(f"PDF is encrypted and cannot be opened: {path}")

    title = None
    if reader.metadata is not None and reader.metadata.title:
        title = normalize_text(str(reader.metadata.title))
    page_count = len(reader.pages)

    pages: list[DocumentPage] = []
    for index, pdf_page in enumerate(reader.pages):
        page_number = index + 1
        error = None
        try:
            raw_text = pdf_page.extract_text() or ""
        except Exception as exc:  # malformed content stream on one page
            logger.warning("Text extraction failed on %s page %d: %s", path.name, page_number, exc)
            raw_text, error = "", f"{type(exc).__name__}: {exc}"

        text = normalize_text(raw_text)

        try:
            image_count = len(pdf_page.images)
        except Exception:
            image_count = 0

        char_count = len(text)
        has_text = char_count >= MIN_TEXT_CHARS
        try:
            page_label = reader.page_labels[index]
        except Exception:
            page_label = str(page_number)

        metadata = {
            "file_name": path.name,
            "title": title,
            "page_count": page_count,
            "page_label": page_label,
            "raw_char_count": len(raw_text),
            "char_count": char_count,
            "image_count": image_count,
            "has_text": has_text,
            "needs_ocr": not has_text and image_count > 0,
            "extractor": "pypdf",
            "normalized": True,
        }
        if error:
            metadata["extraction_error"] = error

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata=metadata,
            )
        )

    empty = [p.page for p in pages if not p.metadata["has_text"]]
    if empty:
        logger.info("%s: %d/%d pages without text: %s", path.name, len(empty), page_count, empty)
    return pages


# --- Normalization -----------------------------------------------------------

# Zero-width / invisible characters that only break tokens and string matching.
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"), None)
# Line/page separators that should become plain newlines.
_LINE_BREAKS = re.compile(r"\r\n?|[\x0b\x0c\x85]")
# Remaining control characters (keep \n and \t).
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
# Any horizontal whitespace run (tabs, NBSP, thin spaces, ...).
_HSPACE = re.compile(r"[^\S\n]+")
# Markdown escapes left by the PDF generator: "\_", "\#", "\{", "\[" ...
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>~\"'])")
# Chapter opener with a drop cap: "B\n3\nINSTALLATION ...\nefore you ..."
_DROP_CAP = re.compile(
    r"\A([A-Z])\n(\d+)\n((?:[^\n]*[A-Z][^a-z\n]*\n)+?)([a-z])"
)
# Compound word wrapped at its hyphen: "free-\ntier" -> "free-tier".
_HYPHEN_WRAP = re.compile(r"(\w)-\n(?=\w)")
_SENTENCE_END = re.compile(r"[.!?:][\"'”’)\]]?$")


def _fix_drop_cap(text: str) -> str:
    """Re-attach a chapter's drop-cap letter and put the heading on its own line."""

    def repl(m: re.Match) -> str:
        letter, number, heading, first = m.groups()
        heading = " ".join(heading.split())
        # A drop-cap "A" is almost always the article ("A writer ..."), other
        # letters are the start of a word ("B" + "efore" -> "Before").
        joiner = " " if letter == "A" else ""
        return f"{number}\n{heading}\n\n{letter}{joiner}{first}"

    return _DROP_CAP.sub(repl, text, count=1)


def _restore_paragraph_breaks(text: str) -> str:
    """Insert blank lines where a wrapped paragraph visibly ends.

    pypdf emits one line per visual line and almost never a blank line, so a
    paragraph-based chunker sees each page as one block. A line that ends a
    sentence and is clearly shorter than the page's text width is treated as
    the last line of a paragraph. Line breaks themselves are kept, so lists,
    code and tables keep their layout.
    """
    lines = text.split("\n")
    lengths = sorted(len(line) for line in lines if line)
    if len(lengths) < 5:
        return text
    width = lengths[int(len(lengths) * 0.9) - 1]

    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            line
            and nxt
            and _SENTENCE_END.search(line)
            and len(line) < 0.9 * width
        ):
            out.append("")
    return "\n".join(out)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding useful structure.

    - NFKC (ligatures, full-width forms, compatibility spaces), drop invisible
      and control characters, unify line endings.
    - Remove Markdown escape backslashes left in the PDF ("\\_" -> "_").
    - Collapse horizontal whitespace, strip each line.
    - Repair chapter drop caps, restore paragraph breaks, re-join compound
      words wrapped at a hyphen.
    - Keep single newlines (lists, code, tables) and at most one blank line
      between paragraphs.

    Typographic quotes and dashes are kept; evaluation folds them when matching.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    text = _LINE_BREAKS.sub("\n", text)
    text = _CONTROL.sub("", text)
    text = _MD_ESCAPE.sub(r"\1", text)

    text = "\n".join(_HSPACE.sub(" ", line).strip() for line in text.split("\n"))
    text = text.strip()

    text = _fix_drop_cap(text)
    text = _restore_paragraph_breaks(text)
    text = _HYPHEN_WRAP.sub(r"\1-", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
