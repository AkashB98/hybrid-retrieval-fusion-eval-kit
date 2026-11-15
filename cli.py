"""CLI: generate/evaluate/report/check/demo/serve.

check gates on the stated floors (not on who wins -- the winner is
reported, never gated):
  - every method's numbers are computed for every query
  - RRF hybrid and router are both reported
  - every recall / bridge-support value is inside [0, 1]
exit 0 = pass, 1 = gate failure, 2 = data error.
check emits a JSON summary with fields: queries, documents, k,
methods (list), winner_by_recall_at_k, winner_by_bridge_support_at_k,
passed (bool), verdict ("pass"|"fail").
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from generator import generate_corpus
from metrics import full_report

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def cmd_generate(args):
    docs, queries = generate_corpus()
    out = {"documents": docs, "queries": queries}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_evaluate(args):
    report = full_report(k=args.k)
    slim = {name: {"recall_at_k": m["recall_at_k"],
                   "bridge_support_at_k": m["bridge_support_at_k"],
                   "by_class": m["by_class"]}
            for name, m in report["methods"].items()}
    print(json.dumps(slim, indent=2, sort_keys=True))
    return 0


def cmd_report(args):
    print(json.dumps(full_report(k=args.k), indent=2, sort_keys=True))
    return 0


def check_summary(report):
    methods = report.get("methods", {})
    n_queries = report.get("corpus", {}).get("queries", 0)
    ok = True
    for name in ("bm25", "tfidf_expanded", "rrf_hybrid", "router"):
        m = methods.get(name)
        if m is None or m.get("queries") != n_queries or n_queries == 0:
            ok = False
            continue
        for key in ("recall_at_k", "bridge_support_at_k"):
            if not (0.0 <= m[key] <= 1.0):
                ok = False
        for cls in m.get("by_class", {}).values():
            for key in ("recall_at_k", "bridge_support_at_k"):
                if not (0.0 <= cls[key] <= 1.0):
                    ok = False
    return {
        "queries": n_queries,
        "documents": report.get("corpus", {}).get("documents", 0),
        "k": report.get("k"),
        "methods": sorted(methods),
        "winner_by_recall_at_k": report.get("winner", {}).get("by_recall_at_k"),
        "winner_by_bridge_support_at_k": report.get("winner", {}).get(
            "by_bridge_support_at_k"),
        "passed": bool(ok),
        "verdict": "pass" if ok else "fail",
    }


def cmd_check(args):
    try:
        report = full_report(k=args.k)
        if report["corpus"]["queries"] == 0:
            raise ValueError("empty corpus")
    except Exception as exc:
        print(json.dumps({"error": str(exc), "verdict": "error"}, sort_keys=True))
        return EXIT_ERROR
    s = check_summary(report)
    print(json.dumps(s, indent=2, sort_keys=True))
    return EXIT_OK if s["passed"] else EXIT_FAIL


def cmd_demo(args):
    import demo
    return demo.main()


def make_server(port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, payload, code=200):
            body = json.dumps(payload, sort_keys=True).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send({"status": "ok"})
            elif self.path == "/report":
                self._send(full_report())
            else:
                self._send({"error": "not found"}, 404)

    return HTTPServer(("127.0.0.1", port), H)


def cmd_serve(args):
    srv = make_server(args.port)
    print(f"serving on 127.0.0.1:{args.port}")
    srv.serve_forever()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="hybridretrieval")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=cmd_generate)
    for name, fn in (("evaluate", cmd_evaluate), ("report", cmd_report),
                     ("check", cmd_check)):
        sp = sub.add_parser(name)
        sp.add_argument("--k", type=int, default=5)
        sp.set_defaults(fn=fn)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    sp = sub.add_parser("serve")
    sp.add_argument("--port", type=int, default=8765)
    sp.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
