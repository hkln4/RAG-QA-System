"""Ground-truth labels: questions with evidence passages located by page.

Evidence is labelled as (document, 1-based PDF page, verbatim text) rather
than as chunk ids, because chunk boundaries change with chunk size, which is
one of the experiment variables. A retrieved chunk counts as relevant when it
overlaps an evidence span on the same page.

    python rag_eval/labels.py   # check every evidence passage against the corpus
"""
import json
import sys
import unicodedata
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
CORPUS_DIR = Path(__file__).parent / "corpus"


def normalize(text):
    """Lowercase alphanumerics only, plus a map back to raw indices.

    PDF extraction splits words at line ends ("inte-\\ngrated"), inserts stray
    spaces ("G OVERN") and keeps ligatures ("ﬁ"); comparing only NFKC-folded
    letters and digits makes evidence matching robust to all three.
    """
    chars, index = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).lower():
            if c.isalnum():
                chars.append(c)
                index.append(i)
    return "".join(chars), index


def locate(page_text, evidence_text):
    """Raw (start, end) span of evidence_text in page_text, or None."""
    norm_page, index = normalize(page_text)
    norm_evidence, _ = normalize(evidence_text)
    if not norm_evidence:
        return None
    pos = norm_page.find(norm_evidence)
    if pos < 0:
        return None
    return index[pos], index[pos + len(norm_evidence) - 1] + 1


def load_questions(path=QUESTIONS_PATH):
    return json.loads(Path(path).read_text())


def load_pages(corpus_dir=CORPUS_DIR):
    """{(file name, 1-based page): page text} as PyPDFLoader extracts it."""
    from langchain_community.document_loaders import PyPDFLoader

    pages = {}
    for pdf in sorted(Path(corpus_dir).glob("*.pdf")):
        for doc in PyPDFLoader(str(pdf)).load():
            pages[(pdf.name, doc.metadata["page"] + 1)] = doc.page_content
    return pages


def evidence_locations(ev):
    """Every (doc, page) holding this passage: its primary page plus `also_on`.

    The same text can legitimately appear twice, e.g. AI 600-1 quotes AI RMF
    subcategories as table headers; a chunk from either copy answers the question.
    """
    return [(ev["doc"], ev["page"])] + [(a["doc"], a["page"]) for a in ev.get("also_on", [])]


def validate(questions, pages):
    """Errors for evidence missing from a listed page; warnings for unlisted copies elsewhere."""
    errors, warnings = [], []
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")
    for q in questions:
        if not q.get("evidence"):
            errors.append(f"{q['id']}: no evidence")
        for ev in q.get("evidence", []):
            listed = evidence_locations(ev)
            for key in listed:
                if key not in pages:
                    errors.append(f"{q['id']}: no page {key}")
                elif locate(pages[key], ev["text"]) is None:
                    errors.append(f"{q['id']}: evidence not found on {key}: {ev['text'][:60]!r}")
            unlisted = [k for k, text in pages.items() if k not in listed and locate(text, ev["text"])]
            if unlisted:
                warnings.append(f"{q['id']}: evidence also appears on {unlisted}; add them to also_on")
    return errors, warnings


if __name__ == "__main__":
    questions = load_questions()
    errors, warnings = validate(questions, load_pages())
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("error:", e)
    print(f"{len(questions)} questions, {len(errors)} errors, {len(warnings)} warnings")
    sys.exit(1 if errors else 0)
