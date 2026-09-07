"""Persistent HTTP service for the G1 DEX3 ACT policy."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path

from infer_g1_dex3_act import CAMERAS, G1ActModel, _group_stats, load_observation

MAX_BODY = 5 * 1024 * 1024


class _Handler(BaseHTTPRequestHandler):
    model: G1ActModel

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(
            200,
            {
                "ready": True,
                "model_digest": self.model.model_digest,
                "action_shape": [100, 28],
                "cameras": list(CAMERAS),
                "language_conditioned": False,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send(404, {"error": "not found"})
            return
        length_header = self.headers.get("Content-Length")
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        try:
            length = int(length_header) if length_header is not None else -1
        except ValueError:
            length = -1
        if content_type != "application/octet-stream" or length < 1 or length > MAX_BODY:
            self._send(400, {"error": "expected application/octet-stream body up to 5 MiB"})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete request body")
            observation = load_observation(io.BytesIO(raw))
            actions, seconds = self.model.infer(observation)
        except (ValueError, OSError, EOFError) as exc:
            self._send(400, {"error": str(exc)})
            return
        self._send(
            200,
            {
                "actions": actions.tolist(),
                "inference_seconds": seconds,
                "model_digest": self.model.model_digest,
                "groupstats": _group_stats(actions),
            },
        )

    def log_message(self, format: str, *args) -> None:
        return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args(argv)
    _Handler.model = G1ActModel(args.checkpoint, args.device)
    server = HTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
