import json
import os
import subprocess
import sys
import threading
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cli import check_summary, make_server  # noqa: E402
from metrics import full_report  # noqa: E402


def run_cli(*argv):
    return subprocess.run([sys.executable, os.path.join(ROOT, "cli.py"), *argv],
                          capture_output=True, text=True, cwd=ROOT, timeout=60)


class TestCLI(unittest.TestCase):
    def test_check_passes_exit_0(self):
        r = run_cli("check")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        s = json.loads(r.stdout)
        self.assertTrue(s["passed"])
        self.assertEqual(s["methods"],
                         ["bm25", "router", "rrf_hybrid", "tfidf_expanded"])

    def test_report_contains_all_methods_and_classes(self):
        r = run_cli("report")
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(report["corpus"]["documents"], 50)
        for m in report["methods"].values():
            self.assertEqual(sorted(m["by_class"]),
                             ["multi_hop_bridge", "named_entity", "paraphrase"])

    def test_generate_is_deterministic(self):
        a, b = run_cli("generate"), run_cli("generate")
        self.assertEqual(a.stdout, b.stdout)

    def test_check_summary_fields(self):
        s = check_summary(full_report())
        for field in ("queries", "documents", "k", "methods",
                      "winner_by_recall_at_k", "passed", "verdict"):
            self.assertIn(field, s)

    def test_serve_round_trip(self):
        server = make_server(0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=10) as r:
                self.assertEqual(json.loads(r.read()), {"status": "ok"})
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/report",
                                        timeout=10) as r:
                served = json.loads(r.read())
            self.assertEqual(served["corpus"]["queries"], 30)
        finally:
            server.shutdown()
            server.server_close()


class TestNoNetwork(unittest.TestCase):
    def test_core_modules_do_not_import_network_or_env(self):
        for fname in ("generator.py", "retrieval.py", "metrics.py"):
            with open(os.path.join(ROOT, fname)) as fh:
                src = fh.read()
            for banned in ("urllib", "socket", "requests", "os.environ",
                           "openai", "http.client"):
                self.assertNotIn(banned, src, f"{fname} mentions {banned}")


if __name__ == "__main__":
    unittest.main()
