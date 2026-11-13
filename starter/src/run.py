from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pypdf import PdfReader

from .chunking import by_paragraph, fixed_size
from .embeddings import ENV_PATH, Embedder, make_embedder
from .evaluation import (
    coherence_positions, coherence_sheet, grounding_sheet, normalize_for_match,
    summarize, validate_question,
)
from .evaluation import evaluate_query
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = STARTER_DIR.parent
DEFAULT_PDF = REPO_DIR / "data" / "master-claude-code-and-cowork.pdf"
DEFAULT_QUESTIONS = REPO_DIR / "benchmark" / "questions.json"
DEFAULT_RESULTS = STARTER_DIR / "results"

DOCUMENT_ID = "master-claude-code-cowork-2026"
MIN_CONTENT_PAGES = 30
TOP_K = 3

CHUNK_A = {"chunking": "fixed_size", "size": 500, "overlap": 50}
CHUNK_B = {"chunking": "by_paragraph", "max_size": 1000, "overlap": 0}
STRATEGIES = {
    "A": {**CHUNK_A, "index": "A", "filter": False},
    "B": {**CHUNK_B, "index": "B", "filter": False},
    "C": {**CHUNK_B, "index": "B", "filter": "question.filter when present, else none"},
}

SECTIONS: list[tuple[int, str, str]] = [
    (1, "front_matter", "Cover"),
    (2, "front_matter", "Title Page"),
    (3, "front_matter", "Copyright"),
    (4, "front_matter", "Epigraph"),
    (5, "front_matter", "Contents"),
    (6, "preface", "Preface: The AI Coworker Has Arrived"),
    (10, "ch01_agentic_leap", "1. The Agentic Leap — From Chat to Action"),
    (15, "ch02_ecosystem", "2. The Ecosystem — Models, Plans, and Platforms"),
    (21, "ch03_installation", "3. Installation and First Launch"),
    (28, "ch04_claude_code_fundamentals", "4. Claude Code Fundamentals — The Agentic CLI"),
    (39, "ch05_everyday_claude_code", "5. Everyday Tasks with Claude Code"),
    (48, "ch06_mcp", "6. MCP — Connecting Claude Code to the World"),
    (61, "ch07_skills_hooks_automation", "7. Skills, Hooks, and Automation"),
    (76, "ch08_ide_cicd", "8. Claude Code in the IDE and CI/CD"),
    (85, "ch09_cowork_essentials", "9. Cowork Essentials"),
    (95, "ch10_everyday_cowork", "10. Everyday Tasks with Cowork"),
    (104, "ch11_connectors_plugins", "11. Connectors and Plugins — Extending Cowork"),
    (113, "ch12_global_instructions", "12. Global Instructions and Personal Configuration"),
    (128, "ch13_safety_security", "13. Safety, Security, and What Can Go Wrong"),
    (135, "ch14_usage_costs", "14. Usage Limits, Costs, and Efficiency"),
    (142, "ch15_config_hierarchy", "15. The Configuration Hierarchy"),
    (149, "ch16_agent_teams", "16. Agent Teams and Parallel Workflows"),
    (158, "ch17_professions", "17. Cowork for Specific Professions"),
    (167, "ch18_cowork_office", "18. Cowork and Microsoft Office"),
    (177, "ch19_non_coders_advanced", "19. Claude Code for Non-Coders — Advanced"),
    (182, "ch20_workflows", "20. Twelve Complete Workflows from Start to Finish"),
    (192, "ch21_choosing_tool", "21. Choosing the Right Tool for the Job"),
    (196, "ch22_future", "22. The Future of Agentic AI"),
    (200, "a_request", "A Request"),
    (201, "appendix_a_cli_reference", "Appendix A: Claude Code CLI Reference"),
    (205, "appendix_b_cowork_commands", "Appendix B: Quick Reference — Cowork Slash Commands"),
    (208, "appendix_c_claude_md_templates", "Appendix C: CLAUDE.md Templates"),
    (214, "appendix_d_troubleshooting", "Appendix D: Troubleshooting"),
]
NON_CONTENT_SECTIONS = {"front_matter"}

log = logging.getLogger("day06")


# ---- 1. corpus ------------------------------------------------------------------
def check_sections_against_outline(pdf: Path) -> None:
    """Fail if the manifest's start pages do not match the PDF outline."""
    reader = PdfReader(pdf)
    outline = {
        reader.get_destination_page_number(item) + 1: item.title
        for item in reader.outline if not isinstance(item, list)
    }
    for start, slug, title in SECTIONS:
        if start == 1:  # cover has no outline entry
            continue
        if start not in outline:
            raise RuntimeError(f"section {slug!r} starts on page {start}, not in PDF outline")
        if not outline[start].startswith(title.split("—")[0].split(";")[0].strip()[:20]):
            raise RuntimeError(f"page {start}: outline title {outline[start]!r} != {title!r}")


def section_for(page: int) -> tuple[str, str]:
    current = SECTIONS[0]
    for entry in SECTIONS:
        if entry[0] <= page:
            current = entry
    return current[1], current[2]


def load_corpus(pdf: Path) -> tuple[list[DocumentPage], dict[str, Any]]:
    check_sections_against_outline(pdf)
    pages = load_pdf(pdf, document_id=DOCUMENT_ID, source=pdf.name)

    year = None
    for page in pages[:6]:
        match = re.search(r"Copyright\s*©\s*(\d{4})", page.text)
        if match:
            year = int(match.group(1))
            break

    for page in pages:
        slug, title = section_for(page.page)
        page.metadata.update({"section": slug, "section_title": title})
        if year is not None:
            page.metadata["year"] = year

    no_text = [p.page for p in pages if not p.metadata["has_text"]]
    front = [p.page for p in pages if p.metadata["section"] in NON_CONTENT_SECTIONS]
    content = [p for p in pages
               if p.metadata["has_text"] and p.metadata["section"] not in NON_CONTENT_SECTIONS]
    if len(content) < MIN_CONTENT_PAGES:
        raise RuntimeError(f"only {len(content)} content pages; need >= {MIN_CONTENT_PAGES}")
    log.info("Loaded %d physical pages, %d content pages (excluded no-text %s, front matter %s)",
             len(pages), len(content), no_text, front)

    info = {
        "physical_pages": len(pages),
        "content_pages": len(content),
        "excluded_no_text_pages": no_text,
        "excluded_front_matter_pages": front,
        "year": year,
        "year_source": "copyright page" if year else None,
        "section_source": "PDF outline (bookmarks), checked at run time",
    }
    return content, info


# ---- 2. benchmark -------------------------------------------------------------
def load_questions(path: Path, pages: list[DocumentPage]) -> tuple[list[dict[str, Any]], list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_page = {(p.document_id, p.page): p for p in pages}
    sections = {slug for _, slug, _ in SECTIONS}
    questions, warnings = [], []

    for item in raw:
        q = dict(item["question"] if "question" in item else item)
        accepted = q.pop("accepted_pages", None)
        if accepted:
            q["gold_page"] = sorted({int(q["gold_page"]), *map(int, accepted)})
        validate_question(q)

        gold_pages = q["gold_page"] if isinstance(q["gold_page"], list) else [q["gold_page"]]
        texts = []
        for page_no in gold_pages:
            page = by_page.get((q["gold_source_id"], int(page_no)))
            if page is None:
                raise ValueError(f"{q['id']}: gold page {q['gold_source_id']} p.{page_no} not in corpus")
            texts.append(normalize_for_match(page.text))
        if not any(normalize_for_match(q["evidence_quote"]) in t for t in texts):
            warnings.append(f"{q['id']}: evidence_quote not found verbatim on gold page(s) {gold_pages}")

        for key, value in (q.get("filter") or {}).items():
            if key == "section" and value not in sections:
                raise ValueError(f"{q['id']}: filter section {value!r} does not exist")
        questions.append(q)

    ids =[q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    for w in warnings:
        log.warning(w)
    log.info("Loaded %d questions (%d dev / %d test, %d with filter)", len(questions),
             sum(q["split"] == "dev" for q in questions),
             sum(q["split"] == "test" for q in questions),
             sum(bool(q.get("filter")) for q in questions))
    return questions, warnings


# ---- 3. chunking ---------------------------------------------------------------
def build_chunks(pages: list[DocumentPage]) -> dict[str, list[Chunk]]:
    chunks_a = [c for p in pages for c in fixed_size(p, CHUNK_A["size"], CHUNK_A["overlap"])]
    chunks_b = [c for p in pages for c in by_paragraph(p, CHUNK_B["max_size"], CHUNK_B["overlap"])]
    return {"A": chunks_a, "B": chunks_b}


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    sizes = [len(c.text) for c in chunks]
    return {
        "chunk_count": len(chunks),
        "total_chars": sum(sizes),
        "approx_tokens": sum(sizes) // 4,
        "min_chars": min(sizes),
        "median_chars": statistics.median(sizes),
        "mean_chars": round(statistics.mean(sizes), 1),
        "max_chars": max(sizes),
    }


# ---- 4. embed + store ------------------------------------------------------------
def ingest(store: PgVectorStore, embedder: Embedder,
           chunks_by_index: dict[str, list[Chunk]]) -> dict[str, Any]:
    store.create_schema()
    stats = {}
    for index, chunks in chunks_by_index.items():
        removed_docs = store.prune_documents(index, {c.document_id for c in chunks})
        if removed_docs:
            log.info("Index %s: removed %d chunks of documents no longer in the corpus",
                     index, removed_docs)
        result = store.index_chunks(index, chunks, embedder, log=log.info)
        stats[index] = {**result.__dict__, "removed_stale_documents": removed_docs,
                        "stored_rows": store.count(index)}
    return stats


# ---- 5. retrieve + evaluate -------------------------------------------------------
def retrieve_and_evaluate(store: PgVectorStore, embedder: Embedder,
                          questions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    query_vectors = {q["id"]: embedder.embed_query(q["question"]) for q in questions}
    embed_seconds = round(time.perf_counter() - started, 2)

    traces = []
    for name, cfg in STRATEGIES.items():
        for q in questions:
            applied = (q.get("filter") or None) if cfg["filter"] else None
            hits = store.search(cfg["index"], query_vectors[q["id"]], top_k=TOP_K, filters=applied)
            trace = evaluate_query(q, hits)
            trace.update({
                "strategy": name,
                "index": cfg["index"],
                "applied_filter": applied,
                "filter_note": (
                    "C: no filter in question, ran unfiltered" if name == "C" and not applied else None
                ),
            })
            traces.append(trace)
        log.info("Strategy %s: retrieved top-%d for %d questions", name, TOP_K, len(questions))
    return traces, embed_seconds


# ---- 6. outputs --------------------------------------------------------------------
def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_manual_csv(path: Path, rows: list[dict[str, Any]], score_columns: tuple[str, ...]) -> Path:
    """Write a manual-scoring sheet without overwriting scores someone already filled in."""
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            filled = any(
                (row.get(col) or "").strip()
                for row in csv.DictReader(f) for col in score_columns
            )
        if filled:
            path = path.with_name(path.stem + ".new" + path.suffix)
            log.warning("Manual scores exist in %s; writing fresh sheet to %s", path.stem, path.name)
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
                                 for k, v in row.items()})
    return path


def summary_markdown(summary: dict[str, Any], ingest_stats: dict[str, Any],
                     chunk_info: dict[str, Any]) -> str:
    lines = [
        "# Day 06 retrieval summary",
        "",
        "| Strategy | Split | N | Hit@1 | Evidence Hit@3 | Filter compliant | Mean top-1 score |",
        "|---|---|---|---|---|---|---|",
    ]
    for strategy, splits in summary.items():
        for split in ("dev", "test", "all"):
            s = splits.get(split)
            if not s:
                continue
            compliant = (f"{s['filter_compliant']}/{s['filter_questions']}"
                         if s["filter_questions"] else "N/A")
            lines.append(
                f"| {strategy} | {split} | {s['n']} | {s['hit_at_1']}/{s['n']} ({s['hit_at_1_rate']:.0%}) "
                f"| {s['hit_at_3']}/{s['n']} ({s['hit_at_3_rate']:.0%}) | {compliant} "
                f"| {s['mean_top1_score']} |"
            )
    lines += ["", "## Chunks", "", "| Index | Chunks | Median chars | Approx tokens | Embedded this run | Ingest s |",
              "|---|---|---|---|---|---|"]
    for index, info in chunk_info.items():
        ing = ingest_stats.get(index, {})
        lines.append(f"| {index} | {info['chunk_count']} | {info['median_chars']} | {info['approx_tokens']} "
                     f"| {ing.get('embedded', 'skipped')} | {ing.get('seconds', 'skipped')} |")
    lines += [
        "",
        "C uses the same chunks and vectors as B (index `B`) and only adds the metadata filter "
        "when the question has one; questions without a filter run unfiltered under C.",
        "Hit@3 is an automatic proxy; coherence and grounding still need the manual sheets.",
    ]
    return "\n".join(lines) + "\n"


def setup_logging(results: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Drop per-request INFO logs from the HTTP / Gemini clients; keep their warnings.
    quiet = ("httpx", "httpcore", "google_genai")

    def keep(record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or not record.name.startswith(quiet)

    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(results / "run.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        handler.addFilter(keep)
        root.addHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 06 pipeline: ingest -> chunk -> embed -> pgvector -> retrieve -> evaluate")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--no-ingest", action="store_true",
                        help="skip embedding/storing chunks; reuse vectors already in pgvector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    setup_logging(args.results)
    load_dotenv(ENV_PATH)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(f"DATABASE_URL is missing; set it in {ENV_PATH}")
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 1-2. corpus + benchmark
    pages, corpus_info = load_corpus(args.pdf)
    questions, question_warnings = load_questions(args.questions, pages)

    # 3. chunking
    chunks_by_index = build_chunks(pages)
    chunk_info = {index: chunk_stats(chunks) for index, chunks in chunks_by_index.items()}
    for index, info in chunk_info.items():
        log.info("Index %s: %s", index, info)

    # 4. embed + store
    embedder = make_embedder()
    store = PgVectorStore(database_url, embedder.dimensions)
    if args.no_ingest:
        ingest_stats = {index: {"skipped": True, "stored_rows": store.count(index)}
                        for index in chunks_by_index}
        for index, chunks in chunks_by_index.items():
            if ingest_stats[index]["stored_rows"] != len(chunks):
                raise RuntimeError(f"index {index}: {ingest_stats[index]['stored_rows']} stored rows "
                                   f"but {len(chunks)} chunks; run without --no-ingest")
    else:
        ingest_stats = ingest(store, embedder, chunks_by_index)

    # 5. retrieve + evaluate
    traces, embed_query_seconds = retrieve_and_evaluate(store, embedder, questions)
    summary = summarize(traces)

    # 6. outputs
    results = args.results
    with (results / "traces.jsonl").open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    write_json(results / "summary.json", summary)
    (results / "summary.md").write_text(summary_markdown(summary, ingest_stats, chunk_info), encoding="utf-8")
    write_json(results / "ingest_stats.json",
               {index: {**chunk_info[index], **ingest_stats[index]} for index in chunks_by_index})

    positions = coherence_positions(pages, count=10)
    coherence_file = write_manual_csv(
        results / "manual_coherence.csv",
        coherence_sheet(positions, chunks_by_index),
        ("coherence_score_0_2", "reason"),
    )
    grounding_file = write_manual_csv(
        results / "manual_grounding.csv",
        grounding_sheet(traces, per_strategy=5),
        ("grounding_score_0_2", "reason", "top1_metadata_correct_y_n",
         "top2_metadata_correct_y_n", "top3_metadata_correct_y_n"),
    )

    write_json(results / "run_config.json", {
        "run_started_utc": started_utc,
        "pdf": str(args.pdf),
        "questions": str(args.questions),
        "document_id": DOCUMENT_ID,
        "embedding_model": embedder.model_name,
        "dimensions": embedder.dimensions,
        "embedding_task_types": {"documents": "RETRIEVAL_DOCUMENT", "queries": "RETRIEVAL_QUERY"},
        "vector_store": "PostgreSQL + pgvector, HNSW index, cosine distance (approximate)",
        "strategies": STRATEGIES,
        "top_k": TOP_K,
        "corpus": corpus_info,
        "sections": [{"start_page": s, "section": slug, "title": t} for s, slug, t in SECTIONS],
        "question_count": len(questions),
        "question_warnings": question_warnings,
        "embed_query_seconds": embed_query_seconds,
        "manual_sheets": {"coherence": coherence_file.name, "grounding": grounding_file.name},
    })

    for strategy, splits in summary.items():
        s = splits["all"]
        log.info("%s: Hit@1 %d/%d, Evidence Hit@3 %d/%d", strategy,
                 s["hit_at_1"], s["n"], s["hit_at_3"], s["n"])
    log.info("Wrote results to %s", results)


if __name__ == "__main__":
    main()
