"""Serve the results dashboard locally.

Usage (from the starter/ folder):
    python scripts/serve_results.py            # http://localhost:8000/starter/dashboard/
    python scripts/serve_results.py --port 9000 --no-browser

The server's root is the repository root, so the dashboard can read both
starter/results/ and artifacts/experiment_log.csv. Files are re-read on every page
load: re-run src/run.py and refresh the browser to see new numbers.

The dashboard can save manual scores back to disk. Only the files listed in
WRITABLE can be written, and only by requests carrying the dashboard's header.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = STARTER_DIR.parent
WRITABLE = {
    "manual_coherence.csv": STARTER_DIR / "results" / "manual_coherence.csv",
    "manual_grounding.csv": STARTER_DIR / "results" / "manual_grounding.csv",
    "experiment_log.csv": REPO_DIR / "artifacts" / "experiment_log.csv",
}
MAX_BODY = 5 * 1024 * 1024


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".jsonl": "application/json; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".log": "text/plain; charset=utf-8",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _reply(self, code: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path != "/api/save":
            return self._reply(404, "not found")
        # A custom header forces a CORS preflight, which this server never approves,
        # so other websites open in the browser cannot write these files.
        if self.headers.get("X-Dashboard") != "1":
            return self._reply(403, "missing dashboard header")
        name = (parse_qs(url.query).get("file") or [""])[0]
        target = WRITABLE.get(name)
        if target is None:
            return self._reply(400, f"file not writable: {name}")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            return self._reply(400, "bad body size")
        data = self.rfile.read(length)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return self._reply(400, "body must be UTF-8")
        target.write_bytes(data)
        print(f"Saved {target}")
        return self._reply(200, f"saved {name}")

    def log_message(self, fmt: str, *args) -> None:  # keep the console quiet
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    handler = functools.partial(DashboardHandler, directory=str(REPO_DIR))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://localhost:{args.port}/starter/dashboard/"
        print(f"Results dashboard: {url}  (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
