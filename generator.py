"""Seeded, deterministic synthetic corpus generator.

Corpus design (no tuning-to-win: classes are planted by construction,
methods are then measured honestly on them):

- named_entity (10 queries, 1 relevant doc each): the query uses an exact
  rare name + organisation token that appears verbatim in the target doc.
  Keyword retrieval should favour these.
- paraphrase (10 queries, 1 relevant doc each): the target doc uses only
  canonical vocabulary; the query uses only synonyms/concept terms from
  EXPANSION_MAP, with zero canonical overlap. Raw keyword retrieval has
  nothing to match; the expanded-vocabulary method can map the query
  back to the canonical terms.
- multi_hop_bridge (10 queries, 2 relevant docs each): doc A carries a
  named endpoint, doc B carries a canonical concept, and A and B share a
  unique bridge token (which is NOT in the query). The query mentions
  the named endpoint (exact) plus a synonym for doc B's concept, so a
  single-method retriever that only does one of {exact, expanded}
  matching tends to find only one hop.
- filler docs (10): unrelated text sharing only generic filler words
  with no query, so they should never be retrieved on a positive score.

Document order is fixed: named docs, paraphrase docs, filler docs,
then bridge pairs (A0, B0, A1, B1, ...). Generation uses no randomness
at all beyond a seeded shuffle of filler words, so two calls always
return identical data.
"""
import random

SEED = 42

# synonym / concept term -> canonical term. This is the stated expansion
# map for the TF-IDF method. It is a stand-in for embeddings, not
# embeddings: it only knows the pairs listed here.
EXPANSION_MAP = {
    # paraphrase triples
    "automobile": "car", "fix": "repair", "price": "cost",
    "physician": "doctor", "sickness": "illness", "medication": "medicine",
    "residence": "house", "yard": "garden", "doorway": "entrance",
    "instructor": "teacher", "academy": "school", "examination": "exam",
    "attorney": "lawyer", "tribunal": "court", "judgment": "verdict",
    "cook": "chef", "formula": "recipe", "taste": "flavor",
    "vocalist": "singer", "melody": "song", "performance": "concert",
    "grower": "farmer", "harvest": "crop", "earth": "soil",
    "aviator": "pilot", "aircraft": "airplane", "airstrip": "runway",
    "writer": "author", "book": "novel", "section": "chapter",
    # bridge second-hop pairs
    "lamp": "lantern", "port": "harbor", "pasture": "meadow",
    "gorge": "canyon", "fiddle": "violin", "fortress": "castle",
    "wasteland": "desert", "woodland": "forest", "isle": "island",
    "peak": "mountain",
}

_NAMES = ["Zelindra", "Korvath", "Mirellen", "Thandor", "Quinara",
          "Brevik", "Solmara", "Daskin", "Yelvara", "Ostrik"]
_ORGS = ["nexalab", "quantive", "helvion", "orbitek", "zentara",
         "velcora", "nimbleq", "ostrova", "keldera", "pravion"]
_TOPICS = ["solar", "ocean", "climate", "forest", "river"]

_PARAPHRASE_TRIPLES = [
    ("car", "repair", "cost"),
    ("doctor", "illness", "medicine"),
    ("house", "garden", "entrance"),
    ("teacher", "school", "exam"),
    ("lawyer", "court", "verdict"),
    ("chef", "recipe", "flavor"),
    ("singer", "song", "concert"),
    ("farmer", "crop", "soil"),
    ("pilot", "airplane", "runway"),
    ("author", "novel", "chapter"),
]
# inverse lookup used only to build queries: canonical -> synonym
_SYNONYM_FOR = {v: k for k, v in EXPANSION_MAP.items()}

_BRIDGE_NAMES = ["Aldren", "Bexley", "Corvin", "Damaris", "Elowen",
                 "Fintan", "Giselle", "Haldor", "Isolde", "Joren"]
_BRIDGE_TOKENS = ["zorlink", "velmark", "quinpath", "brevnode", "solchain",
                  "dasgate", "yelnexus", "osthub", "fenlink", "galcross"]
_BRIDGE_CANONICALS = ["lantern", "harbor", "meadow", "canyon", "violin",
                      "castle", "desert", "forest", "island", "mountain"]

_FILLER_TOPICS = ["baking", "cycling", "painting", "fishing", "hiking",
                  "knitting", "rowing", "skiing", "surfing", "bowling"]


def generate_corpus():
    """Return (docs, queries). docs: [{doc_id, text}];
    queries: [{query_id, qclass, query, relevant}] with relevant sorted."""
    rng = random.Random(SEED)
    docs, queries = [], []

    def add_doc(text):
        doc_id = f"doc-{len(docs):03d}"
        docs.append({"doc_id": doc_id, "text": text})
        return doc_id

    # named_entity docs
    for i, (name, org) in enumerate(zip(_NAMES, _ORGS)):
        topic = _TOPICS[i % len(_TOPICS)]
        low = name.lower()
        doc_id = add_doc(
            f"{low} leads {org} research on {topic}. {low} published a study "
            f"about {topic} methods and {topic} data. the team at {org} "
            f"works with {low} on experiments and field notes.")
        queries.append({"query_id": f"q-named-{i:02d}", "qclass": "named_entity",
                        "query": f"{name} {org} {topic}", "relevant": [doc_id]})

    # paraphrase docs: canonical vocabulary only, never the synonyms
    for i, (c1, c2, c3) in enumerate(_PARAPHRASE_TRIPLES):
        doc_id = add_doc(
            f"the {c1} and the {c2} and the {c3} are discussed in detail. "
            f"{c1} {c2} {c3} notes cover {c1} practice and {c2} practice "
            f"and {c3} practice for beginners and specialists.")
        s1, s2, s3 = (_SYNONYM_FOR[c] for c in (c1, c2, c3))
        queries.append({"query_id": f"q-para-{i:02d}", "qclass": "paraphrase",
                        "query": f"{s1} {s2} {s3}", "relevant": [doc_id]})

    # filler docs: generic words only, no query terms
    filler_words = ["general", "notes", "methods", "data", "study",
                    "detailed", "review", "practice", "beginners", "field"]
    for topic in _FILLER_TOPICS:
        words = filler_words[:]
        rng.shuffle(words)
        add_doc(f"{topic} {' '.join(words)} and {topic} activities for "
                f"local groups during the season.")

    # multi_hop_bridge doc pairs
    for i, (bname, bridge, canon) in enumerate(
            zip(_BRIDGE_NAMES, _BRIDGE_TOKENS, _BRIDGE_CANONICALS)):
        low = bname.lower()
        a_id = add_doc(
            f"{low} discovered {bridge} during an expedition. {low} and "
            f"{bridge} are recorded together in the expedition log.")
        b_id = add_doc(
            f"{bridge} relates to the {canon} in this account. the {canon} "
            f"and {bridge} and the {canon} are described in detail.")
        queries.append({"query_id": f"q-bridge-{i:02d}",
                        "qclass": "multi_hop_bridge",
                        "query": f"{bname} {_SYNONYM_FOR[canon]}",
                        "relevant": sorted([a_id, b_id])})

    return docs, queries
