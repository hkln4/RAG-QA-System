from rag_eval.metrics import covers, evidence_spans, score_retrieval

SPAN = ("a.pdf", 1, 100, 200)  # 100-character evidence span


def test_covers_requires_half_of_the_evidence_span():
    assert covers(("a.pdf", 1, 0, 150), SPAN)        # contains 50 of 100 characters
    assert not covers(("a.pdf", 1, 0, 149), SPAN)    # 49 of 100
    assert covers(("a.pdf", 1, 120, 180), SPAN)      # chunk inside a long span, 60 of 100
    assert not covers(("a.pdf", 2, 0, 1000), SPAN)   # other page
    assert not covers(("b.pdf", 1, 0, 1000), SPAN)   # other document


def test_evidence_spans_include_also_on_copies():
    pages = {("a.pdf", 1): "xx needle yy", ("b.pdf", 4): "needle"}
    q = {"evidence": [{"doc": "a.pdf", "page": 1, "text": "needle", "also_on": [{"doc": "b.pdf", "page": 4}]}]}
    assert evidence_spans(q, pages) == [[("a.pdf", 1, 3, 9), ("b.pdf", 4, 0, 6)]]


def test_score_retrieval_single_evidence():
    spans = [[SPAN]]
    ranked = [("a.pdf", 9, 0, 10), ("a.pdf", 1, 90, 210), ("a.pdf", 1, 0, 10)]
    s = score_retrieval(ranked, spans, ks=(1, 3))
    assert s == {"hit@1": 0.0, "recall@1": 0.0, "hit@3": 1.0, "recall@3": 1.0, "mrr": 0.5}


def test_score_retrieval_multi_hop_needs_every_item_for_full_recall():
    spans = [[SPAN], [("b.pdf", 2, 0, 50)]]
    ranked = [("a.pdf", 1, 100, 200), ("c.pdf", 1, 0, 10), ("b.pdf", 2, 0, 60)]
    s = score_retrieval(ranked, spans, ks=(1, 3))
    assert s["hit@1"] == 1.0 and s["recall@1"] == 0.5
    assert s["recall@3"] == 1.0 and s["mrr"] == 1.0


def test_score_retrieval_alternative_locations_count_once():
    spans = [[SPAN, ("b.pdf", 5, 0, 100)]]
    s = score_retrieval([("b.pdf", 5, 0, 100)], spans, ks=(1,))
    assert s == {"hit@1": 1.0, "recall@1": 1.0, "mrr": 1.0}


def test_score_retrieval_nothing_relevant():
    s = score_retrieval([("z.pdf", 1, 0, 10)], [[SPAN]], ks=(1,))
    assert s == {"hit@1": 0.0, "recall@1": 0.0, "mrr": 0.0}
