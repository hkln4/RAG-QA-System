"""Evaluate one RAG configuration on the NIST eval set.

    python rag_eval/download_corpus.py      # once
    python -m rag_eval.run                  # baseline: the app's current defaults
    python -m rag_eval.run --name chunk_500 --chunk-size 500 --chunk-overlap 100
    python -m rag_eval.experiments          # every quality experiment, one comparison table

Retrieval metrics (hit@k, recall@k, MRR) cover the answerable questions;
RAGAS faithfulness and answer relevancy score the generated answers with a
separate judge model; unanswerable questions measure refusals. Results go to
rag_eval/results/<name>.json and .md, and to MLflow when a server is reachable.
"""
import argparse
import json
import math
import os
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")  # ragas and the legacy google SDK emit deprecation noise on import
# RAGAS sends usage analytics by default; evaluation data should stay local
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from rag_eval.cache import Cache, with_retry
from rag_eval.labels import CORPUS_DIR, load_questions
from rag_eval.metrics import evidence_spans, score_retrieval
from core.retrieval import CANDIDATES  # per-list depth for hybrid fusion and the reranker's candidate pool
from rag_eval.retrievers import BM25, LocalEmbeddings, Reranker, reciprocal_rank_fusion
from rag_pipeline import (
    ANSWER_MODEL,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    PROMPT_TEMPLATE,
    format_docs,
    load_pdfs,
    split_documents,
)

# A different vendor from the answer model, so the judge does not grade its own
# answers (the earlier flash-lite self-judge scored a correct answer 0). Haiku 5.5
# is the cheapest Claude model: ~$0.03 of judge calls per configuration (measured
# with the free token-counting endpoint). Thinking is off: it would bill extra
# output tokens and a RAGAS judge only fills in short JSON.
JUDGE_MODEL = "claude-haiku-5-5"
JUDGE_PRICE_PER_MTOK = (0.10, 0.50)  # input, output; prompts under 100k tokens
# The Gemini embedding free tier (1000 requests a day) is also spent by answer
# relevancy, so its question-similarity embeddings are computed locally instead.
RELEVANCY_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
JUDGE_BUDGET_USD = float(os.getenv("JUDGE_BUDGET_USD", "0.30"))  # total across runs
KS = (1, 3, 5, 10)
REFUSAL = "this information is not available"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass(frozen=True)
class Config:
    name: str = "baseline"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = EMBEDDING_MODEL  # "models/..." is Gemini; anything else is sentence-transformers
    retrieval: str = "dense"  # "dense" or "hybrid" (dense + BM25, fused with RRF)
    reranker: str | None = None  # cross-encoder that reorders the top CANDIDATES dense chunks
    k: int = 5  # chunks passed to the LLM; /query's default
    answer_model: str = ANSWER_MODEL
    # The app answers at 0.3; experiments use 0 so differences between
    # configurations are not mixed with sampling noise.
    temperature: float = 0.0


def chunk_ref(doc):
    m = doc.metadata
    return (Path(m["source"]).name, m["page"] + 1, m["start_index"], m["start_index"] + len(doc.page_content))


def embed(texts, kind, model, embeddings, cache):
    """Cached embeddings; documents go in batches of 100 (the API maximum)."""
    keys = [Cache.key("embedding", model, kind, t) for t in texts]
    vectors = [cache.get(k) for k in keys]
    missing = [i for i, v in enumerate(vectors) if v is None]
    for start in range(0, len(missing), 100):
        batch = missing[start:start + 100]
        if kind == "document":
            fresh = with_retry(embeddings.embed_documents, [texts[i] for i in batch])
        else:
            fresh = [with_retry(embeddings.embed_query, texts[i]) for i in batch]
        for i, v in zip(batch, fresh):
            cache.set(keys[i], v)
            vectors[i] = v
    if missing:
        print(f"  embedded {len(missing)} {kind}s ({len(texts) - len(missing)} cached)")
    return vectors


def make_embeddings(model):
    if model.startswith("models/"):
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=GOOGLE_API_KEY)
    return LocalEmbeddings(model)


class BudgetExceeded(RuntimeError):
    pass


class JudgeSpend(BaseCallbackHandler):
    """Adds up the judge's billed tokens from each response and keeps the running
    total in the cache, so the budget holds across runs, not just within one."""

    KEY = Cache.key("spend", JUDGE_MODEL)

    def __init__(self, cache):
        self.cache = cache
        self.total = cache.get(self.KEY) or {"input_tokens": 0, "output_tokens": 0, "usd": 0.0}

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for g in generations:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.total["input_tokens"] += usage.get("input_tokens", 0)
                self.total["output_tokens"] += usage.get("output_tokens", 0)
        price_in, price_out = JUDGE_PRICE_PER_MTOK
        self.total["usd"] = (self.total["input_tokens"] * price_in + self.total["output_tokens"] * price_out) / 1e6
        self.cache.set(self.KEY, self.total)

    def check(self):
        # checked before each question rather than raised mid-call: RAGAS retries
        # failed calls, so an exception inside one would be retried and cost more
        if self.total["usd"] >= JUDGE_BUDGET_USD:
            raise BudgetExceeded(f"judge spend ${self.total['usd']:.4f} reached the ${JUDGE_BUDGET_USD:.2f} budget")


def make_judge(spend):
    from langchain_anthropic import ChatAnthropic
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms.base import LangchainLLMWrapper

    llm = ChatAnthropic(model=JUDGE_MODEL, max_tokens=4096, thinking={"type": "disabled"},
                        max_retries=4, callbacks=[spend])
    # bypass_n: answer_relevancy asks for 3 completions at once; one call each is
    # supported by every provider and keeps the judge comparable across them.
    # bypass_temperature: RAGAS sets temperature on each call, which Haiku 5.5
    # rejects with a 400 ("temperature is deprecated for this model").
    return (LangchainLLMWrapper(llm, bypass_n=True, bypass_temperature=True),
            LangchainEmbeddingsWrapper(LocalEmbeddings(RELEVANCY_EMBEDDING_MODEL)))


def ragas_scores(question, answer, contexts, judge, cache):
    key = Cache.key("ragas", JUDGE_MODEL, RELEVANCY_EMBEDDING_MODEL, question, answer, contexts)
    cached = cache.get(key)
    if cached is not None:
        return cached
    from datasets import Dataset
    from ragas import RunConfig, evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    llm, embeddings = judge
    result = evaluate(
        Dataset.from_dict({"question": [question], "answer": [answer], "contexts": [contexts]}),
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=4, max_retries=10, max_wait=90, timeout=600),
        show_progress=False,
        raise_exceptions=False,
    )
    row = result.to_pandas().iloc[0]
    scores = {m: float(row[m]) for m in ("faithfulness", "answer_relevancy")}
    if not any(math.isnan(v) for v in scores.values()):  # failed judgements are retried on the next run
        cache.set(key, scores)
    return scores


def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return sum(values) / len(values) if values else None


def summarize(rows, retrieval_only=False):
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    groups = {
        "all": answerable,
        "paraphrased": [r for r in answerable if r["paraphrased"]],
        "not_paraphrased": [r for r in answerable if not r["paraphrased"]],
        "identifier": [r for r in answerable if r["type"] == "identifier"],
        "multi_hop": [r for r in answerable if r["type"] == "multi-hop"],
    }
    retrieval_keys = list(answerable[0]["retrieval"])
    summary = {"retrieval": {}, "generation": {}, "refusal": {}}
    for name, group in groups.items():
        summary["retrieval"][name] = {"n": len(group), **{k: mean([r["retrieval"][k] for r in group]) for k in retrieval_keys}}
    if retrieval_only:
        return summary
    # Faithfulness skips refusals: "not available" makes no claim to check, and the
    # judge scored identical refusals 1.0 and 0.0. Refusals are measured below.
    # Answer relevancy keeps them, since not answering is a real relevancy failure.
    scored = {"faithfulness": [r for r in answerable if not r["refused"]], "answer_relevancy": answerable}
    for metric, group in scored.items():
        values = [r["ragas"][metric] for r in group]
        summary["generation"][metric] = mean(values)
        summary["generation"][f"{metric}_n"] = sum(v is not None and not math.isnan(v) for v in values)
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    summary["refusal"]["unanswerable_refused"] = mean([float(r["refused"]) for r in unanswerable])
    summary["refusal"]["answerable_refused"] = mean([float(r["refused"]) for r in answerable])
    return summary


def to_markdown(result):
    cfg, s = result["config"], result["summary"]
    fmt = lambda v: "–" if v is None else f"{v:.3f}"
    lines = [
        f"# RAG eval: {cfg['name']}",
        "",
        f"{result['timestamp']} · {result['n_questions']} questions · {result['n_chunks']} chunks · "
        f"chunk {cfg['chunk_size']}/{cfg['chunk_overlap']} · k={cfg['k']} · "
        f"embeddings `{cfg['embedding_model']}` · retrieval {cfg['retrieval']} · reranker {cfg['reranker']} · "
        f"answers `{cfg['answer_model']}` (T={cfg['temperature']}) · "
        f"judge `{result['judge_model']}`",
        "",
        "## Retrieval",
        "",
        "| Questions | n | hit@1 | hit@5 | recall@5 | recall@10 | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in s["retrieval"].items():
        lines.append(f"| {name} | {m['n']} | {fmt(m['hit@1'])} | {fmt(m['hit@5'])} | {fmt(m['recall@5'])} | "
                     f"{fmt(m['recall@10'])} | {fmt(m['mrr'])} |")
    g, rf = s["generation"], s["refusal"]
    if not g:
        return "\n".join(lines + ["", "Retrieval only: no answers generated.", ""])
    lines += [
        "",
        "## Generation",
        "",
        "| Faithfulness | Answer relevancy | Unanswerable refused | Answerable refused |",
        "|---|---|---|---|",
        f"| {fmt(g['faithfulness'])} (n={g['faithfulness_n']}) | {fmt(g['answer_relevancy'])} "
        f"(n={g['answer_relevancy_n']}) | {fmt(rf['unanswerable_refused'])} | {fmt(rf['answerable_refused'])} |",
        "",
    ]
    return "\n".join(lines)


def log_to_mlflow(result):
    # same fail-fast settings as the app's tracker: an unreachable server must not stall the run
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    import mlflow

    metrics = {}
    for section, values in result["summary"].items():
        for name, v in values.items():
            if isinstance(v, dict):
                metrics.update({f"{section}.{name}.{k}".replace("@", "_at_"): x for k, x in v.items() if x is not None})
            elif v is not None:
                metrics[f"{section}.{name}"] = v
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
        mlflow.set_experiment("RAG Evaluation")
        with mlflow.start_run(run_name=result["config"]["name"]):
            mlflow.log_params({**result["config"], "judge_model": result["judge_model"]})
            mlflow.log_metrics(metrics)
        print("logged to MLflow")
    except Exception as e:
        print(f"MLflow logging skipped: {str(e)[:120]}")


def run(cfg, retrieval_only=False):
    started = time.time()
    cache = Cache()
    questions = load_questions()

    documents = load_pdfs(str(CORPUS_DIR) + "/")
    pages = {(Path(d.metadata["source"]).name, d.metadata["page"] + 1): d.page_content for d in documents}
    chunks = split_documents(documents, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    texts = [c.page_content for c in chunks]

    embeddings = make_embeddings(cfg.embedding_model)
    vectors = embed(texts, "document", cfg.embedding_model, embeddings, cache)
    metadatas = [{**c.metadata, "chunk_id": i} for i, c in enumerate(chunks)]
    store = FAISS.from_embeddings(list(zip(texts, vectors)), embeddings, metadatas=metadatas)
    query_vectors = embed([q["question"] for q in questions], "query", cfg.embedding_model, embeddings, cache)
    bm25 = BM25(texts) if cfg.retrieval == "hybrid" else None
    reranker = Reranker(cfg.reranker) if cfg.reranker else None

    def retrieve(question, qvec):
        """Top max(KS) chunk ids; dense alone is the baseline, the others add one step each."""
        depth = CANDIDATES if (bm25 or reranker) else max(KS)
        ranked = [d.metadata["chunk_id"] for d, _ in store.similarity_search_with_score_by_vector(qvec, k=depth)]
        if bm25:
            ranked = reciprocal_rank_fusion([ranked, bm25.rank(question, depth)], n=depth)
        if reranker:
            ranked = reranker.rerank(question, ranked, texts)
        return ranked[:max(KS)]

    if not retrieval_only:
        chain = (ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                 | ChatGoogleGenerativeAI(model=cfg.answer_model, google_api_key=GOOGLE_API_KEY,
                                          temperature=cfg.temperature, max_retries=0)
                 | StrOutputParser())
        spend = JudgeSpend(cache)
        judge = make_judge(spend)

    rows = []
    for i, (q, qvec) in enumerate(zip(questions, query_vectors), 1):
        ranked = [chunks[j] for j in retrieve(q["question"], qvec)]
        row = {"id": q["id"], "type": q["type"], "paraphrased": q.get("paraphrased", False),
               "retrieved": [list(chunk_ref(d)) for d in ranked]}
        if q["type"] != "unanswerable":
            row["retrieval"] = score_retrieval([chunk_ref(d) for d in ranked], evidence_spans(q, pages), KS)
        if not retrieval_only:
            context_docs = ranked[:cfg.k]
            inputs = {"context": format_docs(context_docs), "question": q["question"]}
            key = Cache.key("answer", cfg.answer_model, cfg.temperature, PROMPT_TEMPLATE, inputs)
            answer = cache.get(key)
            if answer is None:
                answer = with_retry(chain.invoke, inputs)
                cache.set(key, answer)
            row["answer"], row["refused"] = answer, REFUSAL in answer.lower()
            if q["type"] != "unanswerable":
                spend.check()
                row["ragas"] = ragas_scores(q["question"], answer, [d.page_content for d in context_docs], judge, cache)
            print(f"[{i}/{len(questions)}] {q['id']} {json.dumps(row.get('retrieval', {}).get('hit@5'))} "
                  f"{json.dumps(row.get('ragas'))} refused={row['refused']}", flush=True)
        rows.append(row)

    result = {
        "config": asdict(cfg),
        "retrieval_only": retrieval_only,
        "judge_model": None if retrieval_only else JUDGE_MODEL,
        "relevancy_embedding_model": None if retrieval_only else RELEVANCY_EMBEDDING_MODEL,
        "judge_spend_total": None if retrieval_only else spend.total,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_questions": len(questions),
        "n_chunks": len(chunks),
        "runtime_s": round(time.time() - started),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "summary": summarize(rows, retrieval_only),
        "questions": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{cfg.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    (RESULTS_DIR / f"{cfg.name}.md").write_text(to_markdown(result))
    print(to_markdown(result))
    log_to_mlflow(result)
    return result


def parse_args():
    defaults = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=defaults.name)
    p.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=defaults.chunk_overlap)
    p.add_argument("--embedding-model", default=defaults.embedding_model)
    p.add_argument("--retrieval", choices=("dense", "hybrid"), default=defaults.retrieval)
    p.add_argument("--reranker", default=defaults.reranker)
    p.add_argument("--k", type=int, default=defaults.k)
    p.add_argument("--retrieval-only", action="store_true", help="skip answers and RAGAS (no LLM calls)")
    a = p.parse_args()
    cfg = Config(name=a.name, chunk_size=a.chunk_size, chunk_overlap=a.chunk_overlap,
                 embedding_model=a.embedding_model, retrieval=a.retrieval, reranker=a.reranker, k=a.k)
    return cfg, a.retrieval_only


if __name__ == "__main__":
    cfg, retrieval_only = parse_args()
    run(cfg, retrieval_only)
