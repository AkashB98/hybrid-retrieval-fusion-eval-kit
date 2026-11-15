import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import EXPANSION_MAP, generate_corpus  # noqa: E402


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_corpus(), generate_corpus())

    def test_corpus_shape(self):
        docs, queries = generate_corpus()
        self.assertEqual(len(docs), 50)
        self.assertEqual(len(queries), 30)

    def test_query_classes_balanced(self):
        _, queries = generate_corpus()
        counts = {}
        for q in queries:
            counts[q["qclass"]] = counts.get(q["qclass"], 0) + 1
        self.assertEqual(counts, {"named_entity": 10, "paraphrase": 10,
                                  "multi_hop_bridge": 10})

    def test_bridge_queries_need_two_docs(self):
        docs, queries = generate_corpus()
        ids = {d["doc_id"] for d in docs}
        for q in queries:
            if q["qclass"] == "multi_hop_bridge":
                self.assertEqual(len(q["relevant"]), 2)
            else:
                self.assertEqual(len(q["relevant"]), 1)
            self.assertTrue(set(q["relevant"]) <= ids)

    def test_paraphrase_query_has_no_canonical_overlap(self):
        docs, queries = generate_corpus()
        by_id = {d["doc_id"]: d["text"] for d in docs}
        canonicals = set(EXPANSION_MAP.values())
        for q in queries:
            if q["qclass"] != "paraphrase":
                continue
            doc_text = by_id[q["relevant"][0]]
            for tok in q["query"].split():
                self.assertIn(tok, EXPANSION_MAP)  # query uses synonyms
                self.assertNotIn(tok, doc_text)    # doc never uses them
            self.assertTrue(canonicals & set(doc_text.split()))

    def test_expansion_map_values_are_canonical_targets(self):
        self.assertEqual(EXPANSION_MAP["automobile"], "car")
        self.assertEqual(EXPANSION_MAP["lamp"], "lantern")
        self.assertEqual(len(EXPANSION_MAP), 40)


if __name__ == "__main__":
    unittest.main()
