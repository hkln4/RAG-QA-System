from rag_eval.labels import evidence_locations, locate, normalize, validate


def test_normalize_folds_ligatures_hyphenation_and_spacing():
    assert normalize("proﬁle")[0] == normalize("profile")[0]
    assert normalize("inte-\ngrated")[0] == "integrated"
    assert normalize("G OVERN,")[0] == "govern"


def test_locate_returns_raw_span_despite_line_breaks():
    page = "Intro text. Risk response options can include mitigating, transfer-\nring, avoiding. Next."
    start, end = locate(page, "options can include mitigating, transferring")
    assert page[start:end] == "options can include mitigating, transfer-\nring"


def test_locate_returns_none_when_absent():
    assert locate("some page text", "not on this page") is None


def test_evidence_locations_include_also_on():
    ev = {"doc": "a.pdf", "page": 3, "text": "x", "also_on": [{"doc": "b.pdf", "page": 9}]}
    assert evidence_locations(ev) == [("a.pdf", 3), ("b.pdf", 9)]


def test_validate_flags_missing_evidence_and_unlisted_copies():
    pages = {("a.pdf", 1): "alpha beta gamma", ("a.pdf", 2): "beta gamma again"}
    questions = [
        {"id": "q1", "type": "fact", "evidence": [{"doc": "a.pdf", "page": 1, "text": "beta gamma"}]},
        {"id": "q2", "type": "fact", "evidence": [{"doc": "a.pdf", "page": 2, "text": "delta"}]},
    ]
    errors, warnings = validate(questions, pages)
    assert errors == ["q2: evidence not found on ('a.pdf', 2): 'delta'"]
    assert len(warnings) == 1 and warnings[0].startswith("q1:")


def test_validate_requires_empty_evidence_for_unanswerable():
    pages = {("a.pdf", 1): "alpha"}
    ok = {"id": "q1", "type": "unanswerable", "evidence": []}
    bad = {"id": "q2", "type": "unanswerable", "evidence": [{"doc": "a.pdf", "page": 1, "text": "alpha"}]}
    assert validate([ok], pages) == ([], [])
    assert validate([bad], pages)[0] == ["q2: unanswerable question must not have evidence"]
