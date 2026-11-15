# hybrid-retrieval-fusion-eval-kit

Tests a published retrieval claim on a synthetic corpus: **does a fixed
hybrid — BM25 + a second retriever, merged by Reciprocal Rank Fusion
(RRF) — beat an adaptive router that picks one retrieval method per
query?**

The claim comes from an ablation reported in 2026 (arXiv:2606.21553,
5,000 multi-hop questions): the fixed hybrid beat the adaptive router,
and the router over-fired on named entities. This kit rebuilds that
comparison at small scale, with planted query classes, and reports
whatever actually wins — the corpus was not tuned until RRF won.

Not a duplicate of `rag-chatbot-evals`: that repo evaluates one
retriever plus a refusal gate and answer faithfulness. This kit
compares retrieval *methods* on multi-hop questions and tests a claim.
There is deliberately no answer-generation or answer-support metric
here — that is the other repo's job.

## Verdict from the actual run

On this corpus the published claim **holds against the router, and
only against the router**:

| method | recall@5 | bridge-support@5 |
|---|---|---|
| bm25 | 0.5000 | 0.3333 |
| tfidf_expanded | 1.0000 | 1.0000 |
| **rrf_hybrid** | **1.0000** | **1.0000** |
| router | 0.8333 | 0.6667 |

- Fixed RRF hybrid beats the router overall (recall@5 1.0000 vs
  0.8333; bridge-support@5 1.0000 vs 0.6667), driven entirely by the
  bridge class below — the same failure shape the paper reported.
- Honest caveat: TF-IDF-expanded **alone ties RRF at 1.0000 / 1.0000**
  on this corpus. The run shows fusion beating *routing*, not fusion
  beating the better single method. A corpus where expansion hurts
  some queries (noisier synonyms, denser distractors) is the obvious
  next test, and this kit does not claim that result.

### Per query class (recall@5 / bridge-support@5)

| method | named_entity | paraphrase | multi_hop_bridge |
|---|---|---|---|
| bm25 | 1.0000 / 1.0000 | 0.0000 / 0.0000 | 0.5000 / 0.0000 |
| tfidf_expanded | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| rrf_hybrid | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| router | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.5000 / 0.0000 |

The bridge row is the story: the router routes every bridge query to
BM25 because the query contains a capitalised name, so it finds the
named hop and misses the paraphrased hop on all 10 bridge questions
(bridge-support 0.0000). RRF, which never has to choose, gets both
hops every time.

## POC framing

- **Question:** for a multi-hop RAG proof-of-concept, should retrieval
  be a fixed hybrid or a per-query router?
- **Success criteria:** (1) every method is scored on every query,
  overall and per class; (2) the headline metric is bridge-support@5 —
  *all* documents a multi-hop answer needs inside the top 5 — backed
  by recall@5; (3) the RRF-vs-router comparison is reported per class,
  not just overall, so a class-level failure cannot hide in an average.
- **Thresholds:** bridge-support@5 ≥ 0.90 for a method to be called
  multi-hop-ready on this corpus; a method below 0.50 on any single
  class is disqualified as the sole retriever, however good its
  overall average looks.
- **Gate they feed:** `python3 cli.py check` (exit 0 pass / 1 gate
  failure / 2 data error) verifies the reporting floors — all four
  methods computed for all 30 queries, RRF and router both reported,
  every metric inside [0, 1]. The POC decision rule on top of that
  gate: ship the fixed hybrid if it clears 0.90 bridge-support and
  the router does not; revisit routing only with a router whose
  features can see both hops. On the current run that rule selects
  `rrf_hybrid` (tied with `tfidf_expanded`, which stays the fallback).

## Methods

1. **bm25** — proper BM25 (k1=1.5, b=0.75, standard IDF) over raw
   tokens. Favours exact rare terms: the named-entity class.
2. **tfidf_expanded** — TF-IDF cosine over a synonym/concept-expanded
   vocabulary: a stated 40-pair expansion map (`EXPANSION_MAP` in
   `generator.py`, e.g. `automobile -> car`, `lamp -> lantern`) is
   applied to documents and queries, so paraphrase queries can match
   canonical document terms. **This is a stand-in for embeddings, not
   embeddings** — it knows only the listed pairs, and it is never
   called "dense" in this repo. (Calling plain TF-IDF "dense" would
   just be BM25 with extra steps.)
3. **rrf_hybrid** — fixed Reciprocal Rank Fusion of (1) and (2),
   RRF constant k=60, merging by **rank only**: `rrf_merge()` takes
   ranked id lists, never scores, and a unit test proves the merge is
   invariant to score magnitudes.
4. **router** — picks one method per query from observable features
   of the query string alone: a capitalised token (named entity) →
   BM25; else a synonym/concept term (paraphrase marker) → expanded;
   else BM25. No relevance labels, no ground truth.

## Corpus

Seeded, deterministic, synthetic: 50 documents, 30 queries.

- `named_entity` (10): query uses an exact rare name + organisation
  in the target doc.
- `paraphrase` (10): target doc uses only canonical vocabulary; the
  query uses only expansion-map synonyms, zero canonical overlap.
- `multi_hop_bridge` (10): two relevant docs — one carrying a named
  endpoint, one a canonical concept, sharing a unique bridge token
  that is *not* in the query. The query names endpoint one exactly
  and paraphrases endpoint two.
- 10 filler docs share only generic words with no query.

## Demo (3 commands)

```bash
python3 demo.py                    # full path incl. serve round-trip
python3 -m unittest discover -s tests   # 28 hermetic tests
python3 evals/run_evals.py         # 12 evals -> evals/eval_report.json
```

CLI: `generate` (corpus JSON), `evaluate` (per-method metrics),
`report` (full JSON incl. per-query rows), `check` (gate above),
`demo`, `serve` (stdlib `http.server` on 127.0.0.1, `GET /report`,
`GET /health`). Python stdlib only; no network, no API keys.
`eval_report.json` is byte-stable across runs (sorted keys, no
timestamps).

## Files

| file | purpose |
|---|---|
| `generator.py` | seeded corpus + the stated expansion map |
| `retrieval.py` | BM25, TF-IDF-expanded, rank-only RRF, feature router |
| `metrics.py` | recall@k, bridge-support@k, overall + per-class evaluation |
| `cli.py` | generate/evaluate/report/check/demo/serve |
| `demo.py` | end-to-end demo incl. serve round-trip |
| `evals/run_evals.py` | 12 evals, writes `evals/eval_report.json` |
| `tests/` | 28 hermetic unit tests (incl. RRF rank-not-score test) |

## Honest limits

- Synthetic corpus, small scale (50 docs / 30 queries), planted
  classes: the gaps are large by construction. This tests whether the
  *mechanism* (routing loses a hop; fusion does not) reproduces, not
  the paper's effect size on real questions.
- TF-IDF over an expansion map is a stand-in for embeddings, not
  embeddings. A real dense retriever would also catch paraphrases the
  40-pair map misses — and might blur exact named-entity matches in
  ways this stand-in does not.
- The router is deliberately simple (two observable features). The
  result indicts *this* router design — and any router that keys on
  named-entity presence for bridge queries — not all possible routers.
- As noted in the verdict: expanded-alone ties RRF here, so this run
  is not evidence that fusion beats a strong single method in general.

## Dev-loop honesty notes

- First design merged RRF over *full* rankings including zero-score
  documents; tie-broken early-ID documents would then collect RRF
  credit for queries they never matched and pollute the top 5. Fixed
  by having every retriever return positive scores only, before any
  fusion — and pinned by a test.
- The winner tie-break originally folded the method name into the
  comparison key, which silently picked the alphabetically-last tied
  method. Fixed to a stated rule (first in sorted order) and the tie
  itself is disclosed in the verdict above instead of hidden by it.

## Why this exists

Portfolio project for forward-deployed / applied AI roles: take a
published claim, rebuild it as a small deterministic eval, separate
the headline metric that actually matters for multi-hop questions
(bridge support, not just recall), and report the result that came
out — including the part that flatters the proposed method less.

MIT — see `LICENSE`.
