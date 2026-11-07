"""Download the evaluation corpus: three public-domain NIST publications.

The PDFs are not committed; checksums pin the exact versions the eval set
was labelled against, since page numbers in questions.json depend on them.

    python rag_eval/download_corpus.py
"""
import hashlib
import urllib.request
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"

DOCUMENTS = {
    "nist_ai_rmf.pdf": (
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "7576edb531d9848825814ee88e28b1795d3a84b435b4b797d3670eafdc4a89f1",
    ),
    "nist_ai_600_1.pdf": (
        "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "6e73620ab6b64e90ef2c04bf0e0d6246185a2f4b1b13cab0df494496cff89b6a",
    ),
    "nist_csf_2.pdf": (
        "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
        "3c31f46fee98cac0c4323453e5109291a213b4de7fef8c058af9bf67f717433c",
    ),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    CORPUS_DIR.mkdir(exist_ok=True)
    for name, (url, expected) in DOCUMENTS.items():
        path = CORPUS_DIR / name
        if not path.exists() or sha256(path) != expected:
            print(f"downloading {name}")
            # nvlpubs.nist.gov rejects urllib's default User-Agent with a 403
            request = urllib.request.Request(url, headers={"User-Agent": "rag-qa-system-eval"})
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"{name}: checksum mismatch ({actual}); the source PDF has changed")
        print(f"{name}: ok")


if __name__ == "__main__":
    main()
