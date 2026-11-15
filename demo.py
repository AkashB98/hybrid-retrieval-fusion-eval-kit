"""End-to-end demo incl. serve round-trip. Stdlib only, offline."""
import json
import os
import sys
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cli import check_summary, make_server  # noqa: E402
from generator import generate_corpus  # noqa: E402
from metrics import full_report  # noqa: E402


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=10) as r:
        return r.status, json.loads(r.read())


def main():
    docs, queries = generate_corpus()
    assert (docs, queries) == generate_corpus(), "deterministic"
    print(f"1. corpus: {len(docs)} docs, {len(queries)} queries "
          f"(10 named_entity, 10 paraphrase, 10 multi_hop_bridge)")
    report = full_report()
    for name in sorted(report["methods"]):
        m = report["methods"][name]
        print(f"   {name:<15} recall@5={m['recall_at_k']:.4f} "
              f"bridge_support@5={m['bridge_support_at_k']:.4f}")
    print(f"2. winner by recall@5: {report['winner']['by_recall_at_k']}; "
          f"by bridge_support@5: {report['winner']['by_bridge_support_at_k']}")
    s = check_summary(report)
    assert s["passed"] and s["verdict"] == "pass", s
    print("3. check gate (all methods computed, RRF + router reported): PASS")
    server = make_server(0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        st, health = _get(port, "/health")
        assert health == {"status": "ok"}, health
        st, served = _get(port, "/report")
        assert served == json.loads(json.dumps(report, sort_keys=True)), \
            "served report differs from local report"
        print("4. serve round-trip: GET /report matches local, /health ok")
    finally:
        server.shutdown()
        server.server_close()
    print("demo: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
