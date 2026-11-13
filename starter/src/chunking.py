from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Chunk, DocumentPage

_DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


_STRUCTURE_SEPARATORS = [r"\n\n", r"(?<=[.!?])\s+", r"\n", r"(?<=[;,])\s+", r" ", r""]


def _make_chunk(
    page: DocumentPage, text: str, index: int, strategy: str, start: int, **extra
) -> Chunk:
    return Chunk(
        chunk_id=f"{page.document_id}-p{page.page}-{strategy}-{index}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            **page.metadata,
            "strategy": strategy,
            "chunk_index": index,
            "char_start": start,
            "char_end": start + len(text),
            "chunk_chars": len(text),
            **extra,
        },
    )


def _locate(text: str, pieces: Sequence[str]) -> list[tuple[int, str]]:
    """Find each piece's start offset in text, scanning forward (pieces may overlap)."""
    located: list[tuple[int, str]] = []
    cursor = 0
    for piece in pieces:
        pos = text.find(piece, cursor)
        if pos < 0:  # should not happen with strip-only splitters; search globally
            pos = text.find(piece)
        located.append((max(pos, 0), piece))
        if pos >= 0:
            cursor = pos + 1
    return located


def _split_page(
    page: DocumentPage, splitter: RecursiveCharacterTextSplitter, strategy: str, **extra
) -> list[Chunk]:
    if not page.text.strip():
        return []
    pieces = [p for p in splitter.split_text(page.text) if p.strip()]
    return [
        _make_chunk(page, piece, i, strategy, start, **extra)
        for i, (start, piece) in enumerate(_locate(page.text, pieces))
    ]


def _check_sizes(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")


# --- A: fixed size -----------------------------------------------------------

def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Recursive split with LangChain's default separators into ~size-char
    windows overlapping by up to `overlap` chars (baseline)."""
    _check_sizes(size, overlap)
    splitter = RecursiveCharacterTextSplitter(
        separators=_DEFAULT_SEPARATORS,
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        strip_whitespace=True,
    )
    return _split_page(page, splitter, "fixed", chunk_size=size, chunk_overlap=overlap)


# --- B: paragraph / structure-aware ------------------------------------------

def by_paragraph(page: DocumentPage, max_size: int, overlap: int = 0) -> list[Chunk]:
    """Recursive split that keeps whole paragraphs up to max_size and breaks
    oversized ones at sentence, line, clause, then word boundaries."""
    _check_sizes(max_size, overlap)
    splitter = RecursiveCharacterTextSplitter(
        separators=_STRUCTURE_SEPARATORS,
        is_separator_regex=True,
        keep_separator="end",
        chunk_size=max_size,
        chunk_overlap=overlap,
        length_function=len,
        strip_whitespace=True,
    )
    return _split_page(page, splitter, "paragraph", chunk_size=max_size, chunk_overlap=overlap)
