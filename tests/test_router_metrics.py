import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import generate_corpus  # noqa: E402
from metrics import bridge_support_at_k, evaluate, recall_at_k  # noqa: E402
from retrieval import route_query  # noqa: E402


class TestRouter(unittest.TestCase):
    def test_named_query_routes_to_bm25(self):
        self.assertEqual(route_query("Zelindra nexalab solar"),
                         ("bm25", "named_term_present"))

    def test_paraphrase_query_routes_to_expanded(self):
        self.assertEqual(route_query("automobile fix price"),
                         ("tfidf_expanded", "paraphrase_marker_present"))

    def test_bridge_query_overfires_on_named_term(self):
        # The documented failure mode: a bridge query carrying a
        # capitalised name routes to BM25 even though its second hop
        # needs expansion.
        method, reason = route_query("Aldren lamp")
        self.assertEqual((method, reason), ("bm25", "named_term_present"))

    def test_default_route_is_bm25(self):
        self.assertEqual(route_query("solar data")[0], "bm25")


class TestMetrics(unittest.TestCase):
    def test_recall_at_k(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], ["a", "c"], 2), 0.5)
        self.assertEqual(recall_at_k(["a", "b"], ["a", "b"], 5), 1.0)
        self.assertEqual(recall_at_k(["x"], ["a"], 5), 0.0)

    def test_bridge_support_requires_all_docs(self):
        self.assertEqual(bridge_support_at_k(["a", "b", "c"], ["a", "c"], 3), 1.0)
        self.assertEqual(bridge_support_at_k(["a", "b"], ["a", "c"], 2), 0.0)

    def test_evaluate_shape_and_winner_reported(self):
        docs, queries = generate_corpus()
        report = evaluate(docs, queries)
        self.assertEqual(sorted(report["methods"]),
                         ["bm25", "router", "rrf_hybrid", "tfidf_expanded"])
        for m in report["methods"].values():
            self.assertEqual(m["queries"], 30)
            self.assertTrue(0.0 <= m["recall_at_k"] <= 1.0)
        self.assertIn(report["winner"]["by_recall_at_k"], report["methods"])


if __name__ == "__main__":
    unittest.main()
