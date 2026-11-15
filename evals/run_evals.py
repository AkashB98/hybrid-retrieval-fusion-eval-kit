"""Golden evals. Deterministic; run 3x, md5 of eval_report.json must match.

Cases E1-E7 assert structural gates (all methods computed, RRF and
router reported, ranges valid). Cases E8-E12 assert the actual measured
numbers from the seeded corpus -- whatever they are, they are pinned
here so a silent change in a retriever shows up as an eval failure.
They are NOT gates on who wins: if a future corpus change makes the
router win, update these pins in the same commit and say so in the
README.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from generator import generate_corpus  # noqa: E402
from metrics import full_report  # noqa: E402
from retrieval import route_query  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"eval": name, "pass": bool(ok), "detail": detail})


def main():
    docs, queries = generate_corpus()
    report = full_report()
    m = report["methods"]

    check("E1_corpus_shape",
          report["corpus"] == {"documents": 50, "queries": 30,
                               "queries_by_class": {
                                   "multi_hop_bridge": 10,
                                   "named_entity": 10,
                                   "paraphrase": 10}},
          f"{report['corpus']}")
    check("E2_all_methods_computed",
          all(m[name]["queries"] == 30 for name in
              ("bm25", "tfidf_expanded", "rrf_hybrid", "router")))
    check("E3_rrf_and_router_both_reported",
          "rrf_hybrid" in m and "router" in m)
    check("E4_metrics_in_range",
          all(0.0 <= m[name][key] <= 1.0 for name in m
              for key in ("recall_at_k", "bridge_support_at_k")))
    check("E5_determinism", full_report() == report)
    check("E6_router_overfires_on_bridge_queries",
          all(route_query(q["query"])[0] == "bm25" for q in queries
              if q["qclass"] == "multi_hop_bridge"))
    check("E7_bridge_queries_have_two_relevant",
          all(len(q["relevant"]) == 2 for q in queries
              if q["qclass"] == "multi_hop_bridge"))

    check("E8_bm25_headline",
          m["bm25"]["recall_at_k"] == 0.5
          and m["bm25"]["bridge_support_at_k"] == 0.3333,
          f"recall={m['bm25']['recall_at_k']} "
          f"bridge={m['bm25']['bridge_support_at_k']}")
    check("E9_expanded_headline",
          m["tfidf_expanded"]["recall_at_k"] == 1.0
          and m["tfidf_expanded"]["bridge_support_at_k"] == 1.0,
          f"recall={m['tfidf_expanded']['recall_at_k']} "
          f"bridge={m['tfidf_expanded']['bridge_support_at_k']}")
    check("E10_rrf_headline",
          m["rrf_hybrid"]["recall_at_k"] == 1.0
          and m["rrf_hybrid"]["bridge_support_at_k"] == 1.0,
          f"recall={m['rrf_hybrid']['recall_at_k']} "
          f"bridge={m['rrf_hybrid']['bridge_support_at_k']}")
    check("E11_router_headline",
          m["router"]["recall_at_k"] == 0.8333
          and m["router"]["bridge_support_at_k"] == 0.6667,
          f"recall={m['router']['recall_at_k']} "
          f"bridge={m['router']['bridge_support_at_k']}")
    check("E12_bridge_class_gap_rrf_vs_router",
          m["rrf_hybrid"]["by_class"]["multi_hop_bridge"]["bridge_support_at_k"] == 1.0
          and m["router"]["by_class"]["multi_hop_bridge"]["bridge_support_at_k"] == 0.0,
          "rrf bridge_support=1.0 vs router bridge_support=0.0 on multi_hop_bridge")

    passed = sum(1 for r in RESULTS if r["pass"])
    out = {"k": report["k"], "passed": passed, "total": len(RESULTS),
           "results": RESULTS, "winner": report["winner"],
           "methods_summary": {
               name: {"recall_at_k": mm["recall_at_k"],
                      "bridge_support_at_k": mm["bridge_support_at_k"],
                      "by_class": mm["by_class"]}
               for name, mm in m.items()}}
    (ROOT / "evals" / "eval_report.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"evals: {passed}/{len(RESULTS)} passed")
    for r in RESULTS:
        print(("PASS " if r["pass"] else "FAIL ") + r["eval"] +
              (f"  {r['detail']}" if r["detail"] else ""))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
