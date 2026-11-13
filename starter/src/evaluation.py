"""TODO: evaluate retrieved evidence and preserve per-query traces."""
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from .models import Chunk, DocumentPage

REQUIRED_QUESTION_FIELDS = (
    "id", "split", "question", "expected_answer", "gold_source_id", "gold_page", "evidence_quote",
)
VALID_SPLITS = {"dev", "test"}
COLUMN_FIELDS = {"document_id", "source", "page"}

_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "­": None,
})


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTE_MAP).lower()
    text = re.sub(r"-\s+", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_for_match(text)))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def validate_question(question: dict[str, Any]) -> None:
    missing = [f for f in REQUIRED_QUESTION_FIELDS if question.get(f) in (None, "", [])]
    if missing:
        raise ValueError(f"question {question.get('id')!r} is missing fields: {missing}")
    if question["split"] not in VALID_SPLITS:
        raise ValueError(f"question {question['id']!r} has split {question['split']!r}; use dev or test")
    for alt in question.get("alternatives", []):
        alt_missing = [f for f in ("gold_source_id", "gold_page", "evidence_quote") if not alt.get(f)]
        if alt_missing:
            raise ValueError(f"question {question['id']!r} alternative is missing {alt_missing}")


def gold_options(question: dict[str, Any]) -> list[dict[str, Any]]:
    options = [question, *question.get("alternatives", [])]
    return [
        {
            "source_id": str(opt["gold_source_id"]),
            "pages": {int(p) for p in _as_list(opt["gold_page"])},
            "evidence_quote": opt["evidence_quote"],
        }
        for opt in options
    ]


def _field_value(hit: dict[str, Any], key: str) -> Any:
    if key in COLUMN_FIELDS:
        return hit.get(key)
    return (hit.get("metadata") or {}).get(key)


def _matches_filter(hit: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        allowed = _as_list(expected)
        if _field_value(hit, key) not in allowed:
            return False
    return True


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """TODO: check source/page and enough gold evidence; keep hit@1 and hit@3 trace.

    Accept any documented alternative gold evidence. Do not infer human grounding
    or chunk coherence labels from exact string match.
    """
    validate_question(question)
    options = gold_options(question)
    filters = question.get("filter") or {}

    traced_hits = []
    for rank, hit in enumerate(hits[:3], start=1):
        hit_text = normalize_for_match(hit.get("text", ""))
        hit_tokens = _tokens(hit.get("text", ""))
        best: dict[str, Any] = {"source_match": False, "page_match": False,
                                "evidence_match": False, "evidence_coverage": 0.0, "gold_option": None}
        for index, opt in enumerate(options):
            source_match = opt["source_id"] in (str(hit.get("document_id")), str(hit.get("source")))
            page_match = source_match and int(hit.get("page", -1)) in opt["pages"]
            evidence_tokens = _tokens(opt["evidence_quote"])
            coverage = len(evidence_tokens & hit_tokens) / len(evidence_tokens) if evidence_tokens else 0.0
            evidence_match = normalize_for_match(opt["evidence_quote"]) in hit_text
            candidate = {
                "source_match": source_match, "page_match": page_match,
                "evidence_match": evidence_match, "evidence_coverage": round(coverage, 3),
                "gold_option": index,
            }
            key = (page_match and evidence_match, page_match, source_match, coverage)
            best_key = (best["page_match"] and best["evidence_match"], best["page_match"],
                        best["source_match"], best["evidence_coverage"])
            if key > best_key:
                best = candidate
        traced_hits.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id"),
            "document_id": hit.get("document_id"),
            "source": hit.get("source"),
            "page": hit.get("page"),
            "score": hit.get("score"),
            "metadata": hit.get("metadata"),
            "text": hit.get("text"),
            **best,
            "is_hit": best["page_match"] and best["evidence_match"],
            "filter_match": _matches_filter(hit, filters) if filters else None,
        })

    hit_ranks = [h["rank"] for h in traced_hits if h["is_hit"]]
    first_hit_rank = hit_ranks[0] if hit_ranks else None

    failure = None
    if first_hit_rank is None:
        if not traced_hits:
            failure = "no_results"
        elif not any(h["source_match"] for h in traced_hits):
            failure = "wrong_source"
        elif not any(h["page_match"] for h in traced_hits):
            failure = "wrong_page"
        else:
            failure = "evidence_incomplete"
            for index, opt in enumerate(options):
                on_page = [h for h in traced_hits if h["page_match"] and h["gold_option"] == index]
                union = set().union(*(_tokens(h["text"] or "") for h in on_page)) if on_page else set()
                evidence_tokens = _tokens(opt["evidence_quote"])
                if len(on_page) > 1 and evidence_tokens and evidence_tokens <= union:
                    failure = "evidence_split_across_chunks"
                    break

    return {
        "question_id": question["id"],
        "split": question["split"],
        "type": question.get("type"),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold": [
            {"source_id": o["source_id"], "pages": sorted(o["pages"]), "evidence_quote": o["evidence_quote"]}
            for o in options
        ],
        "filter": filters or None,
        "hits": traced_hits,
        "hit_at_1": first_hit_rank == 1,
        "hit_at_3": first_hit_rank is not None,
        "first_hit_rank": first_hit_rank,
        "filter_compliant": all(h["filter_match"] for h in traced_hits) if filters and traced_hits else None,
        "failure": failure,
        "manual": {
            "grounding_score": None,
            "grounding_reason": None,
            "metadata_correct": None,
            "notes": None,
        },
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """TODO: aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        strategy = trace.get("strategy")
        if not strategy:
            raise ValueError(f"trace {trace.get('question_id')!r} has no strategy")
        groups[(strategy, trace["split"])].append(trace)
        groups[(strategy, "all")].append(trace)

    summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for (strategy, split), items in sorted(groups.items()):
        n = len(items)
        hit1 = sum(t["hit_at_1"] for t in items)
        hit3 = sum(t["hit_at_3"] for t in items)
        filtered = [t for t in items if t.get("filter")]
        applied = [t for t in filtered if t.get("applied_filter")]
        compliant = sum(bool(t["filter_compliant"]) for t in filtered)
        top1_scores = [t["hits"][0]["score"] for t in items if t["hits"]]
        summary[strategy][split] = {
            "n": n,
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "hit_at_1_rate": round(hit1 / n, 4) if n else None,
            "hit_at_3_rate": round(hit3 / n, 4) if n else None,
            "filter_questions": len(filtered),
            "filter_applied": len(applied),
            "filter_compliant": compliant if filtered else None,
            "mean_top1_score": round(sum(top1_scores) / len(top1_scores), 4) if top1_scores else None,
            "failures": dict(Counter(t["failure"] for t in items if t["failure"])),
        }
    return dict(summary)


def coherence_positions(pages: list[DocumentPage], count: int = 10,
                        anchor_chars: int = 80) -> list[dict[str, Any]]:
    content = sorted(
        (p for p in pages if p.metadata.get("has_text", True) and len(p.text) > anchor_chars * 3),
        key=lambda p: (p.document_id, p.page),
    )
    if not content:
        return []
    step = max(len(content) / count, 1)
    chosen = [content[int(i * step)] for i in range(min(count, len(content)))]

    positions = []
    for page in chosen:
        middle = len(page.text) // 2
        start = page.text.rfind(" ", 0, middle) + 1
        end = page.text.find(" ", start + anchor_chars)
        anchor = page.text[start:end if end != -1 else None].strip()
        positions.append({"document_id": page.document_id, "page": page.page, "anchor": anchor})
    return positions


def coherence_sheet(positions: list[dict[str, Any]],
                    chunks_by_strategy: dict[str, list[Chunk]]) -> list[dict[str, Any]]:
    rows = []
    for strategy, chunks in chunks_by_strategy.items():
        for pos in positions:
            anchor = normalize_for_match(pos["anchor"])
            head = anchor[: len(anchor) // 2]
            candidates = [
                c for c in chunks if c.document_id == pos["document_id"] and c.page == pos["page"]
            ]
            match = next((c for c in candidates if anchor in normalize_for_match(c.text)), None)
            if match is None:
                match = next((c for c in candidates if head in normalize_for_match(c.text)), None)
            rows.append({
                "strategy": strategy,
                "document_id": pos["document_id"],
                "page": pos["page"],
                "anchor": pos["anchor"],
                "chunk_id": match.chunk_id if match else None,
                "chunk_text": match.text if match else None,
                "coherence_score_0_2": "",
                "reason": "",
            })
    return rows


def grounding_sheet(traces: list[dict[str, Any]], per_strategy: int = 5) -> list[dict[str, Any]]:
    rows = []
    taken: Counter = Counter()
    for trace in sorted(traces, key=lambda t: (t["strategy"], t["question_id"])):
        if trace["split"] != "test" or taken[trace["strategy"]] >= per_strategy:
            continue
        taken[trace["strategy"]] += 1
        row = {
            "strategy": trace["strategy"],
            "question_id": trace["question_id"],
            "question": trace["question"],
            "expected_answer": trace["expected_answer"],
            "filter": trace.get("applied_filter"),
            "evidence_hit_at_3_proxy": trace["hit_at_3"],
        }
        for hit in trace["hits"]:
            prefix = f"top{hit['rank']}"
            row[f"{prefix}_source_page"] = f"{hit['source']} p.{hit['page']}"
            row[f"{prefix}_text"] = hit["text"]
            row[f"{prefix}_metadata_correct_y_n"] = ""
        row["grounding_score_0_2"] = ""
        row["reason"] = ""
        rows.append(row)
    return rows
