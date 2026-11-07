"""Retrieval metrics over page-and-evidence labels.

Chunks and evidence are both character spans on a page: (doc, page, start, end).
"""
from rag_eval.labels import evidence_locations, locate

# A chunk must contain at least this share of an evidence span to count as
# retrieving it; a chunk that only shares a few boundary characters does not.
MIN_EVIDENCE_COVERAGE = 0.5


def evidence_spans(question, pages):
    """Per evidence item, every (doc, page, start, end) where its text appears."""
    spans = []
    for ev in question["evidence"]:
        item = []
        for doc, page in evidence_locations(ev):
            start, end = locate(pages[(doc, page)], ev["text"])
            item.append((doc, page, start, end))
        spans.append(item)
    return spans


def covers(chunk, span):
    c_doc, c_page, c_start, c_end = chunk
    s_doc, s_page, s_start, s_end = span
    if (c_doc, c_page) != (s_doc, s_page):
        return False
    overlap = min(c_end, s_end) - max(c_start, s_start)
    return overlap >= MIN_EVIDENCE_COVERAGE * (s_end - s_start)


def covered_items(chunk, spans):
    """Indices of the evidence items this chunk retrieves."""
    return {i for i, item in enumerate(spans) if any(covers(chunk, s) for s in item)}


def score_retrieval(ranked_chunks, spans, ks):
    """hit@k, recall@k for each k, and MRR over the whole ranked list.

    recall counts evidence items rather than chunks, so it stays comparable
    across chunk sizes; a multi-hop question needs every item for recall 1.
    """
    per_rank = [covered_items(c, spans) for c in ranked_chunks]
    scores = {}
    for k in ks:
        found = set().union(*per_rank[:k]) if k else set()
        scores[f"hit@{k}"] = float(bool(found))
        scores[f"recall@{k}"] = len(found) / len(spans)
    first = next((rank for rank, items in enumerate(per_rank, 1) if items), None)
    scores["mrr"] = 1 / first if first else 0.0
    return scores
