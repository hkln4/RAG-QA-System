"""Run the quality-axis experiments and write one comparison table.

Each experiment changes one variable of the baseline (the API's defaults).
Retrieval runs make no LLM calls; the baseline also reuses its cached answers
and judge scores. Every run is cached, so rerunning costs only new embeddings.

    python -m rag_eval.experiments   ->  rag_eval/results/experiments.{json,md}
"""
import json
import platform

from rag_eval.cache import QuotaExhausted
from rag_eval.metrics import paired_bootstrap_ci
from rag_eval.run import RESULTS_DIR, Config, run

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# (variable under test, its value) -> config; everything else stays at the baseline
EXPERIMENTS = [
    ("baseline", "–", Config()),
    ("chunk size", "500 / 100", Config(name="chunk_500", chunk_size=500, chunk_overlap=100)),
    ("chunk size", "2000 / 400", Config(name="chunk_2000", chunk_size=2000, chunk_overlap=400)),
    ("embedding", "bge-small-en-v1.5", Config(name="bge_small", embedding_model="BAAI/bge-small-en-v1.5")),
    ("embedding", "bge-base-en-v1.5", Config(name="bge_base", embedding_model="BAAI/bge-base-en-v1.5")),
    ("retrieval", "BM25 + dense (RRF)", Config(name="hybrid", retrieval="hybrid")),
    ("reranker", "MiniLM-L6 cross-encoder", Config(name="rerank", reranker=RERANKER)),
]
PRIMARY = ("recall@5", "mrr")


def per_question(result, metric):
    return {r["id"]: r["retrieval"][metric] for r in result["questions"] if "retrieval" in r}


def compare(results):
    base = results[0][2]
    rows = []
    for variable, value, result in results:
        s = result["summary"]["retrieval"]
        row = {"name": result["config"]["name"], "variable": variable, "value": value,
               "n_chunks": result["n_chunks"],
               **{m: s["all"][m] for m in ("hit@1", "hit@5", "recall@5", "recall@10", "mrr")},
               "identifier_hit@5": s["identifier"]["hit@5"], "paraphrased_hit@5": s["paraphrased"]["hit@5"]}
        for metric in PRIMARY:
            b, c = per_question(base, metric), per_question(result, metric)
            diffs = [c[q] - b[q] for q in b]
            row[f"delta_{metric}"] = sum(diffs) / len(diffs)
            row[f"ci_{metric}"] = paired_bootstrap_ci(diffs)
        rows.append(row)
    return rows


def to_markdown(rows, pending):
    signed = lambda v: f"{v:+.3f}"
    ci = lambda r, m: "–" if r["name"] == "baseline" else f"{signed(r[f'delta_{m}'])} [{signed(r[f'ci_{m}'][0])}, {signed(r[f'ci_{m}'][1])}]"
    lines = [
        "# Quality experiments",
        "",
        f"46 answerable questions; each row changes one variable from the baseline "
        f"(chunk 1000/200, gemini-embedding-001, dense FAISS, no reranker). "
        f"Δ is the mean per-question difference from the baseline with a paired bootstrap 95% CI; "
        f"an interval that contains 0 is not a measurable difference on this set. "
        f"Run on {platform.platform()}.",
        "",
        "| Variable | Value | Chunks | hit@1 | hit@5 | recall@5 | recall@10 | MRR | Δ recall@5 [95% CI] | Δ MRR [95% CI] | Identifier hit@5 | Reworded hit@5 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variable']} | {r['value']} | {r['n_chunks']} | {r['hit@1']:.3f} | {r['hit@5']:.3f} | "
            f"{r['recall@5']:.3f} | {r['recall@10']:.3f} | {r['mrr']:.3f} | {ci(r, 'recall@5')} | {ci(r, 'mrr')} | "
            f"{r['identifier_hit@5']:.3f} | {r['paraphrased_hit@5']:.3f} |")
    for variable, value, reason in pending:
        lines.append(f"| {variable} | {value} | pending: {reason} |" + " |" * 9)
    return "\n".join(lines) + "\n"


def main():
    results, pending = [], []
    for variable, value, cfg in EXPERIMENTS:
        print(f"== {cfg.name}", flush=True)
        try:
            results.append((variable, value, run(cfg, retrieval_only=cfg.name != "baseline")))
        except QuotaExhausted as e:
            # a free-tier daily quota ends one experiment, not the whole sweep; cache keeps its progress
            print(f"   skipped: {e}", flush=True)
            pending.append((variable, value, "daily API quota"))
    rows = compare(results)
    (RESULTS_DIR / "experiments.json").write_text(json.dumps({"rows": rows, "pending": pending}, indent=2) + "\n")
    (RESULTS_DIR / "experiments.md").write_text(to_markdown(rows, pending))
    print(to_markdown(rows, pending))


if __name__ == "__main__":
    main()
