"""Metrics and full evaluation of every method, overall and per class.

Headline metrics (per the cross-check cuts -- no answer-support metric,
that is the shipped rag-chatbot-evals repo's job):

- recall@k: mean fraction of a query's relevant docs inside the top k.
- bridge_support@k: mean, over queries, of 1.0 when ALL relevant docs
  are inside the top k else 0.0. For multi-hop questions this is the
  metric that matters: missing one hop means the answer cannot be
  assembled, even if recall@k looks decent.
"""
from generator import generate_corpus
from retrieval import METHODS, METHOD_NAMES, Index

DEFAULT_K = 5
QUERY_CLASSES = ["named_entity", "paraphrase", "multi_hop_bridge"]


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & set(relevant)) / len(relevant)


def bridge_support_at_k(ranked_ids, relevant, k):
    top = set(ranked_ids[:k])
    return 1.0 if set(relevant) <= top else 0.0


def _round(x):
    return round(x, 4)


def evaluate(docs=None, queries=None, k=DEFAULT_K):
    if docs is None or queries is None:
        docs, queries = generate_corpus()
    index = Index(docs)
    methods = {}
    for name in METHOD_NAMES:
        fn = METHODS[name]
        per_query = []
        for q in queries:
            ranked = [doc_id for doc_id, _ in fn(index, q["query"])]
            per_query.append({
                "query_id": q["query_id"], "qclass": q["qclass"],
                "recall_at_k": _round(recall_at_k(ranked, q["relevant"], k)),
                "bridge_support_at_k": _round(
                    bridge_support_at_k(ranked, q["relevant"], k)),
            })
        by_class = {}
        for cls in QUERY_CLASSES:
            rows = [r for r in per_query if r["qclass"] == cls]
            by_class[cls] = {
                "queries": len(rows),
                "recall_at_k": _round(
                    sum(r["recall_at_k"] for r in rows) / len(rows)) if rows else 0.0,
                "bridge_support_at_k": _round(
                    sum(r["bridge_support_at_k"] for r in rows) / len(rows)) if rows else 0.0,
            }
        methods[name] = {
            "queries": len(per_query),
            "recall_at_k": _round(
                sum(r["recall_at_k"] for r in per_query) / len(per_query)),
            "bridge_support_at_k": _round(
                sum(r["bridge_support_at_k"] for r in per_query) / len(per_query)),
            "by_class": by_class,
            "per_query": per_query,
        }
    # deterministic tie-break: first method in sorted name order wins a tie
    winner_recall = max(METHOD_NAMES, key=lambda m: methods[m]["recall_at_k"])
    winner_bridge = max(
        METHOD_NAMES, key=lambda m: methods[m]["bridge_support_at_k"])
    return {
        "k": k,
        "corpus": {
            "documents": len(docs),
            "queries": len(queries),
            "queries_by_class": {
                cls: sum(1 for q in queries if q["qclass"] == cls)
                for cls in QUERY_CLASSES},
        },
        "methods": methods,
        "winner": {"by_recall_at_k": winner_recall,
                   "by_bridge_support_at_k": winner_bridge},
    }


def full_report(k=DEFAULT_K):
    return evaluate(k=k)
