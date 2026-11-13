from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from .embeddings import Embedder
from .models import Chunk

TABLE = "chunks"
MAX_INDEXED_DIMENSIONS = 2000
COLUMN_FILTERS = {"document_id", "source", "page"}
DEFAULT_METADATA_FILTERS = frozenset({
    "file_name", "title", "page_label", "has_text", "needs_ocr",
    "section", "chapter", "topic", "doc_type", "language",
})


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class IndexStats:
    strategy: str
    embedding_model: str
    total: int = 0          # non-empty chunks received
    embedded: int = 0       # new or changed -> embedded and written
    unchanged: int = 0      # same text + model -> stored vector kept
    skipped_empty: int = 0  # empty text, not stored
    removed: int = 0        # obsolete chunks deleted
    seconds: float = 0.0


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int,
                 metadata_filters: set[str] | frozenset[str] = DEFAULT_METADATA_FILTERS) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if dimensions > MAX_INDEXED_DIMENSIONS:
            raise ValueError(
                f"pgvector HNSW index supports at most {MAX_INDEXED_DIMENSIONS} dimensions, "
                f"got {dimensions}"
            )
        self.database_url = database_url
        self.dimensions = dimensions
        self.metadata_filters = frozenset(metadata_filters)

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, connect_timeout=10)

    # ---- schema -------------------------------------------------------------
    def create_schema(self) -> None:
        table = sql.Identifier(TABLE)
        statements = [
            sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"),
            sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    strategy        TEXT        NOT NULL,
                    chunk_id        TEXT        NOT NULL,
                    document_id     TEXT        NOT NULL,
                    source          TEXT        NOT NULL,
                    page            INTEGER     NOT NULL,
                    text            TEXT        NOT NULL,
                    metadata        JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding       vector({dims}) NOT NULL,
                    content_hash    TEXT,
                    embedding_model TEXT,
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (strategy, chunk_id)
                )
            """).format(table=table, dims=sql.Literal(self.dimensions)),
            # Tables created by an earlier version lack these columns.
            sql.SQL("ALTER TABLE {t} ADD COLUMN IF NOT EXISTS content_hash TEXT").format(t=table),
            sql.SQL("ALTER TABLE {t} ADD COLUMN IF NOT EXISTS embedding_model TEXT").format(t=table),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} USING hnsw (embedding vector_cosine_ops)"
            ).format(name=sql.Identifier(f"{TABLE}_embedding_hnsw"), table=table),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} (strategy, document_id)"
            ).format(name=sql.Identifier(f"{TABLE}_strategy_document"), table=table),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} USING gin (metadata jsonb_path_ops)"
            ).format(name=sql.Identifier(f"{TABLE}_metadata_gin"), table=table),
        ]
        with self._connect() as conn, conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
            cur.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
                (TABLE,),
            )
            row = cur.fetchone()
            if row is None or row[0] != self.dimensions:
                existing = None if row is None else row[0]
                raise RuntimeError(
                    f"table {TABLE!r} stores vector({existing}) but the model produces "
                    f"{self.dimensions} dimensions; use a new table or re-create it"
                )

    # ---- embed + store pipeline ---------------------------------------------
    def index_chunks(self, strategy: str, chunks: Sequence[Chunk], embedder: Embedder,
                     batch_size: int = 100,
                     log: Callable[[str], None] | None = print) -> IndexStats:
        """Embed only new/changed chunks, upsert them in batches and prune obsolete ones.

        Pass ALL chunks of every document being (re)ingested for this strategy:
        stored chunks of those documents that are missing from ``chunks`` are deleted.
        """
        if not strategy:
            raise ValueError("strategy is required")
        if embedder.dimensions != self.dimensions:
            raise ValueError(
                f"embedder produces {embedder.dimensions} dimensions, store expects {self.dimensions}"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._check_unique(chunks)

        started = time.perf_counter()
        model = embedder.model_name
        stats = IndexStats(strategy=strategy, embedding_model=model)

        kept = [c for c in chunks if c.text and c.text.strip()]
        stats.skipped_empty = len(chunks) - len(kept)
        stats.total = len(kept)

        # Decide what needs (re-)embedding.
        stored = self.fingerprints(strategy, [c.chunk_id for c in kept])
        to_embed: list[Chunk] = []
        unchanged: list[Chunk] = []
        for chunk in kept:
            if stored.get(chunk.chunk_id) == (content_hash(chunk.text), model):
                unchanged.append(chunk)
            else:
                to_embed.append(chunk)

        # Embed + write batch by batch, so a failure mid-way keeps finished batches.
        for start in range(0, len(to_embed), batch_size):
            batch = to_embed[start:start + batch_size]
            vectors = embedder.embed_documents([c.text for c in batch])
            self.upsert(strategy, batch, vectors, embedding_model=model, prune=False)
            stats.embedded += len(batch)
            if log:
                log(f"[{strategy}] embedded + stored {stats.embedded}/{len(to_embed)} chunks")

        self._refresh_metadata(strategy, unchanged)
        stats.unchanged = len(unchanged)

        # Prune once over the complete chunk set of each document (not per batch),
        # including documents whose chunks all turned out empty.
        ids_by_document: dict[str, list[str]] = {c.document_id: [] for c in chunks}
        for chunk in kept:
            ids_by_document[chunk.document_id].append(chunk.chunk_id)
        with self._connect() as conn, conn.cursor() as cur:
            stats.removed = self._prune(cur, strategy, ids_by_document)

        stats.seconds = round(time.perf_counter() - started, 2)
        if log:
            log(f"[{strategy}] {stats}")
        return stats

    def fingerprints(self, strategy: str, chunk_ids: list[str]) -> dict[str, tuple[str, str]]:
        """Return {chunk_id: (content_hash, embedding_model)} for chunks already stored."""
        if not chunk_ids:
            return {}
        query = sql.SQL(
            "SELECT chunk_id, content_hash, embedding_model FROM {table} "
            "WHERE strategy = %s AND chunk_id = ANY(%s)"
        ).format(table=sql.Identifier(TABLE))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (strategy, chunk_ids))
            return {cid: (h, m) for cid, h, m in cur.fetchall() if h and m}

    def count(self, strategy: str | None = None) -> int:
        query = sql.SQL("SELECT count(*) FROM {table}").format(table=sql.Identifier(TABLE))
        params: tuple = ()
        if strategy is not None:
            query = query + sql.SQL(" WHERE strategy = %s")
            params = (strategy,)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()[0]

    # ---- low-level writes ---------------------------------------------------
    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]], *,
               embedding_model: str | None = None, prune: bool = True) -> None:
        """Insert or update chunks + vectors keyed by (strategy, chunk_id).

        With ``prune=True`` the call is treated as the complete chunk set of its
        documents and their other stored chunks are deleted. Use ``prune=False`` when a
        document is written over several batches (``index_chunks`` prunes once at the end).
        """
        if not strategy:
            raise ValueError("strategy is required")
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        self._check_unique(chunks)

        ids_by_document: dict[str, list[str]] = defaultdict(list)
        rows = []
        for chunk, vector in zip(chunks, vectors):
            ids_by_document[chunk.document_id].append(chunk.chunk_id)
            rows.append((
                strategy, chunk.chunk_id, chunk.document_id, chunk.source, chunk.page,
                chunk.text, Jsonb(chunk.metadata), self._to_vector_literal(vector),
                content_hash(chunk.text), embedding_model,
            ))

        insert = sql.SQL("""
            INSERT INTO {table}
                (strategy, chunk_id, document_id, source, page, text, metadata, embedding,
                 content_hash, embedding_model)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
            ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                document_id     = EXCLUDED.document_id,
                source          = EXCLUDED.source,
                page            = EXCLUDED.page,
                text            = EXCLUDED.text,
                metadata        = EXCLUDED.metadata,
                embedding       = EXCLUDED.embedding,
                content_hash    = EXCLUDED.content_hash,
                embedding_model = EXCLUDED.embedding_model,
                updated_at      = now()
        """).format(table=sql.Identifier(TABLE))

        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(insert, rows)
            if prune:
                self._prune(cur, strategy, ids_by_document)

    def delete_document(self, strategy: str, document_id: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {table} WHERE strategy = %s AND document_id = %s")
                .format(table=sql.Identifier(TABLE)),
                (strategy, document_id),
            )
            return cur.rowcount

    def prune_documents(self, strategy: str, keep_document_ids: Iterable[str]) -> int:
        """Delete all chunks of documents that are no longer in the corpus."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {table} WHERE strategy = %s AND NOT (document_id = ANY(%s))")
                .format(table=sql.Identifier(TABLE)),
                (strategy, sorted(set(keep_document_ids))),
            )
            return cur.rowcount

    def _refresh_metadata(self, strategy: str, chunks: Sequence[Chunk]) -> None:
        """Update non-vector fields of unchanged chunks; rows already identical are skipped."""
        if not chunks:
            return
        update = sql.SQL("""
            UPDATE {table} SET
                document_id = %s, source = %s, page = %s, metadata = %s, updated_at = now()
            WHERE strategy = %s AND chunk_id = %s
              AND (document_id, source, page, metadata)
                  IS DISTINCT FROM (%s::text, %s::text, %s::integer, %s::jsonb)
        """).format(table=sql.Identifier(TABLE))
        rows = []
        for c in chunks:
            meta = Jsonb(c.metadata)
            rows.append((c.document_id, c.source, c.page, meta, strategy, c.chunk_id,
                         c.document_id, c.source, c.page, meta))
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(update, rows)

    @staticmethod
    def _prune(cur: psycopg.Cursor, strategy: str,
               ids_by_document: dict[str, list[str]]) -> int:
        """Delete stored chunks of the given documents whose IDs are not in the new set."""
        delete_obsolete = sql.SQL(
            "DELETE FROM {table} WHERE strategy = %s AND document_id = %s "
            "AND NOT (chunk_id = ANY(%s))"
        ).format(table=sql.Identifier(TABLE))
        removed = 0
        for document_id, chunk_ids in ids_by_document.items():
            cur.execute(delete_obsolete, (strategy, document_id, chunk_ids))
            removed += cur.rowcount
        return removed

    @staticmethod
    def _check_unique(chunks: Iterable[Chunk]) -> None:
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            seen.add(chunk.chunk_id)

    # ---- search -------------------------------------------------------------
    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL, rank by cosine distance, return top-k.

        Returns chunk ID, text, score (cosine similarity) and metadata. All user-supplied
        values are passed as query parameters; filter keys are checked against an allowlist.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_literal = self._to_vector_literal(query_vector)

        conditions = [sql.SQL("strategy = %s")]
        params: list[Any] = [query_literal, strategy]
        for key, value in (filters or {}).items():
            condition, values = self._build_filter(key, value)
            conditions.append(condition)
            params.extend(values)
        params.extend([query_literal, top_k])

        query = sql.SQL("""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {table}
            WHERE {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """).format(table=sql.Identifier(TABLE), where=sql.SQL(" AND ").join(conditions))

        with self._connect() as conn, conn.cursor() as cur:
            if filters:
                # Keep scanning the HNSW graph until enough rows pass the filter.
                cur.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            cur.execute(query, params)
            return [
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source": source,
                    "page": page,
                    "text": text,
                    "metadata": metadata,
                    "score": float(score),
                }
                for chunk_id, document_id, source, page, text, metadata, score in cur.fetchall()
            ]

    def _build_filter(self, key: str, value: Any) -> tuple[sql.Composable, list[Any]]:
        is_many = isinstance(value, (list, tuple, set, frozenset))
        if is_many and not value:
            raise ValueError(f"filter {key!r} has an empty list of values")

        if key in COLUMN_FILTERS:
            column = sql.Identifier(key)
            if is_many:
                return sql.SQL("{} = ANY(%s)").format(column), [list(value)]
            return sql.SQL("{} = %s").format(column), [value]

        if key in self.metadata_filters:
            options = list(value) if is_many else [value]
            condition = sql.SQL("({})").format(
                sql.SQL(" OR ").join(sql.SQL("metadata @> %s") for _ in options)
            )
            return condition, [Jsonb({key: option}) for option in options]

        allowed = sorted(COLUMN_FILTERS | self.metadata_filters)
        raise ValueError(f"filter {key!r} is not permitted; allowed: {allowed}")

    def _to_vector_literal(self, vector: list[float]) -> str:
        if len(vector) != self.dimensions:
            raise ValueError(f"expected {self.dimensions} dimensions, got {len(vector)}")
        values = []
        for v in vector:
            number = float(v)
            if not math.isfinite(number):
                raise ValueError("vector contains NaN or infinity")
            values.append(repr(number))
        return "[" + ",".join(values) + "]"
