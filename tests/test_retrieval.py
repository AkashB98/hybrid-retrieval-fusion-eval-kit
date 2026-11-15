import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import generate_corpus  # noqa: E402
from retrieval import (Index, bm25_rank, expand_tokens, rrf_merge,  # noqa: E402
                       tfidf_expanded_rank)


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.queries = generate_corpus()
        cls.index = Index(cls.docs)

    def test_bm25_finds_named_doc_first(self):
        q = [x for x in self.queries if x["qclass"] == "named_entity"][0]
        ranked = bm25_rank(self.index, q["query"])
        self.assertEqual(ranked[0][0], q["relevant"][0])

    def test_bm25_misses_paraphrase_doc(self):
        q = [x for x in self.queries if x["qclass"] == "paraphrase"][0]
        ranked = [d for d, _ in bm25_rank(self.index, q["query"])]
        self.assertNotIn(q["relevant"][0], ranked[:5])

    def test_expanded_finds_paraphrase_doc_first(self):
        q = [x for x in self.queries if x["qclass"] == "paraphrase"][0]
        ranked = tfidf_expanded_rank(self.index, q["query"])
        self.assertEqual(ranked[0][0], q["relevant"][0])

    def test_expand_tokens_maps_synonym_to_canonical(self):
        self.assertEqual(expand_tokens(["automobile", "zelindra"]),
                         ["car", "zelindra"])

    def test_rankings_positive_scores_only_and_sorted(self):
        for q in self.queries:
            ranked = bm25_rank(self.index, q["query"])
            self.assertTrue(all(s > 0 for _, s in ranked))
            self.assertEqual(ranked, sorted(ranked, key=lambda kv: (-kv[1], kv[0])))


class TestRRF(unittest.TestCase):
    def test_rrf_merge_is_rank_based_not_score_based(self):
        # Two score scenarios with wildly different magnitudes but the
        # SAME rankings must produce the SAME RRF result, because
        # rrf_merge consumes rankings (id lists), never scores.
        def ranks_from_scores(scores):
            return [d for d, _ in sorted(scores.items(),
                                         key=lambda kv: (-kv[1], kv[0]))]
        scenario_a = ranks_from_scores({"d1": 100.0, "d2": 1.0, "d3": 0.001})
        scenario_b = ranks_from_scores({"d1": 0.51, "d2": 0.49, "d3": 0.48})
        other = ["d2", "d1", "d3"]
        self.assertEqual(scenario_a, scenario_b)
        self.assertEqual(rrf_merge([scenario_a, other]),
                         rrf_merge([scenario_b, other]))

    def test_rrf_rewards_doc_ranked_high_in_both_lists(self):
        merged = rrf_merge([["a", "b", "c"], ["a", "c", "b"]])
        self.assertEqual(merged[0][0], "a")
        self.assertAlmostEqual(merged[0][1], 2.0 / 61.0)

    def test_rrf_doc_in_only_one_list_still_appears(self):
        merged = dict(rrf_merge([["a", "b"], ["c"]]))
        self.assertIn("c", merged)
        self.assertAlmostEqual(merged["c"], 1.0 / 61.0)

    def test_rrf_k_constant_is_60(self):
        import retrieval
        self.assertEqual(retrieval.RRF_K, 60)


if __name__ == "__main__":
    unittest.main()
