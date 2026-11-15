"""Retrieval methods: BM25, TF-IDF over an expanded vocabulary, fixed
Reciprocal Rank Fusion (RRF) hybrid, and an observable-feature router.

Stdlib only. Every method returns a ranked list of (doc_id, score) with
score > 0 only, sorted by (-score, doc_id) so rankings are deterministic.

Honesty labels:
- The "expanded" method is TF-IDF over a synonym/concept-expanded
  vocabulary (generator.EXPANSION_MAP). It is a stand-in that lets
  paraphrase queries match canonical document terms. It is NOT dense
  embeddings and is never called "dense" in this repo.
- RRF merges by RANK only. rrf_merge() takes ranked id lists, never
  scores, so score magnitudes from the two methods cannot leak in.
"""
import math
import re
from collections import Counter

from generator import EXPANSION_MAP

RRF_K = 60  # stated RRF constant (Cormack et al. use 60)

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "on", "in", "at", "to", "by",
    "with", "for", "are", "is", "this", "that", "during", "about",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def expand_tokens(tokens):
    """Map each token to its canonical term if it is in the expansion
    map; canonical and unknown tokens pass through unchanged."""
    return [EXPANSION_MAP.get(t, t) for t in tokens]


class Index:
    """Precomputed per-document statistics for both vocabularies."""

    def __init__(self, docs):
        self.doc_ids = [d["doc_id"] for d in docs]
        self.raw = {d["doc_id"]: tokenize(d["text"]) for d in docs}
        self.expanded = {doc_id: expand_tokens(toks)
                         for doc_id, toks in self.raw.items()}
        self.n = len(docs)

    def _df(self, vocab):
        df = Counter()
        for toks in vocab.values():
            for term in set(toks):
                df[term] += 1
        return df


def bm25_rank(index, query_text):
    """Proper BM25 (k1=1.5, b=0.75) over raw tokens."""
    qtoks = tokenize(query_text)
    if not qtoks:
        return []
    df = index._df(index.raw)
    lengths = {d: len(t) for d, t in index.raw.items()}
    avgdl = sum(lengths.values()) / max(1, index.n)
    scores = {}
    for doc_id, toks in index.raw.items():
        tf = Counter(toks)
        score = 0.0
        for term in qtoks:
            if term not in tf:
                continue
            dfi = df[term]
            idf = math.log(1.0 + (index.n - dfi + 0.5) / (dfi + 0.5))
            freq = tf[term]
            denom = freq + 1.5 * (1.0 - 0.75 + 0.75 * lengths[doc_id] / avgdl)
            score += idf * (freq * (1.5 + 1.0) / denom)
        if score > 1e-12:
            scores[doc_id] = score
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def tfidf_expanded_rank(index, query_text):
    """TF-IDF cosine over the synonym/concept-expanded vocabulary."""
    qtoks = expand_tokens(tokenize(query_text))
    if not qtoks:
        return []
    df = index._df(index.expanded)

    def vec(toks):
        tf = Counter(toks)
        out = {}
        for term, freq in tf.items():
            idf = math.log(1.0 + index.n / (1.0 + df.get(term, 0)))
            out[term] = freq * idf
        norm = math.sqrt(sum(v * v for v in out.values())) or 1.0
        return {t: v / norm for t, v in out.items()}

    qvec = vec(qtoks)
    scores = {}
    for doc_id, toks in index.expanded.items():
        dvec = vec(toks)
        score = sum(qvec.get(t, 0.0) * dvec.get(t, 0.0) for t in qvec)
        if score > 1e-12:
            scores[doc_id] = score
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def rrf_merge(ranked_id_lists, k=RRF_K):
    """Reciprocal Rank Fusion over ranked id lists. Rank-based only:
    the inputs are orderings, not scores. Returns [(doc_id, rrf_score)]
    sorted by (-score, doc_id)."""
    scores = {}
    for ranked in ranked_id_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def rrf_rank(index, query_text):
    bm = [doc_id for doc_id, _ in bm25_rank(index, query_text)]
    ex = [doc_id for doc_id, _ in tfidf_expanded_rank(index, query_text)]
    return rrf_merge([bm, ex])


def route_query(query_text):
    """Observable-feature router. Features are visible in the query
    string alone -- no relevance labels, no corpus ground truth:

    - a capitalised token (a named entity as written by the asker)
      -> BM25
    - else, a token that is a synonym/concept term in EXPANSION_MAP
      (a paraphrase marker) -> TF-IDF expanded
    - else -> BM25 (default)

    Returns (method_name, reason). The known failure mode, by design
    of these features: a bridge query containing a capitalised name
    routes to BM25 even though its second hop needs expansion."""
    raw_tokens = re.findall(r"[A-Za-z0-9]+", query_text)
    if any(t[:1].isupper() for t in raw_tokens):
        return "bm25", "named_term_present"
    lowered = [t.lower() for t in raw_tokens]
    if any(t in EXPANSION_MAP for t in lowered):
        return "tfidf_expanded", "paraphrase_marker_present"
    return "bm25", "default"


def router_rank(index, query_text):
    method, _reason = route_query(query_text)
    if method == "tfidf_expanded":
        return tfidf_expanded_rank(index, query_text)
    return bm25_rank(index, query_text)


METHODS = {
    "bm25": bm25_rank,
    "tfidf_expanded": tfidf_expanded_rank,
    "rrf_hybrid": rrf_rank,
    "router": router_rank,
}
METHOD_NAMES = sorted(METHODS)
