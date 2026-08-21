"""Loopback HTTP bridge for the Lemonade Bench CLI.

The embedded Lemonade 11.6 CLI intermittently reports ``Failed to read
connection`` for otherwise-successful requests sent directly over HTTPS. Keep
the upstream CLI and result format, but let Python handle TLS while the CLI
talks plain HTTP over loopback.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import http.server
import json
import ssl
import subprocess
import sys
import threading
from collections.abc import Sequence
from typing import Any, cast
from urllib.parse import SplitResult, urlsplit

DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 86_400
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "content-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _validated_upstream(value: str) -> SplitResult:
    upstream = urlsplit(value)
    if upstream.scheme not in {"http", "https"} or not upstream.hostname:
        raise ValueError("Lemonade Bench upstream must be an HTTP(S) URL")
    if upstream.username or upstream.password or upstream.query or upstream.fragment:
        raise ValueError(
            "Lemonade Bench upstream cannot contain credentials or query data"
        )
    return upstream


def _connection(upstream: SplitResult, timeout: int) -> http.client.HTTPConnection:
    hostname = upstream.hostname
    if not hostname:
        raise ValueError("Lemonade Bench upstream hostname is required")
    if upstream.scheme == "https":
        return http.client.HTTPSConnection(
            hostname,
            upstream.port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
    return http.client.HTTPConnection(hostname, upstream.port, timeout=timeout)


def _handler(
    upstream: SplitResult, timeout: int
) -> type[http.server.BaseHTTPRequestHandler]:
    class ForwardingHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _forward(self) -> None:
            try:
                content_length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                self.send_error(400, "Invalid Content-Length")
                return
            body = self.rfile.read(content_length) if content_length else None
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in _HOP_BY_HOP_HEADERS
                and key.lower() != "accept-encoding"
                and key.lower() != "host"
            }
            headers["Host"] = upstream.netloc
            headers["Connection"] = "close"
            if body is not None:
                headers["Content-Length"] = str(len(body))
            upstream_path = upstream.path.rstrip("/")
            path = f"{upstream_path}{self.path}" or "/"
            connection = _connection(upstream, timeout)
            try:
                connection.request(self.command, path, body=body, headers=headers)
                response = connection.getresponse()
                response_body = response.read()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() not in _HOP_BY_HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(response_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(response_body)
            except (BrokenPipeError, ConnectionError):
                return
            except Exception as exc:  # pragma: no cover - network-dependent
                payload = json.dumps(
                    {"error": {"message": f"Lemonade Bench proxy failed: {exc}"}}
                ).encode()
                with contextlib.suppress(BrokenPipeError):
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
            finally:
                connection.close()

        do_DELETE = _forward
        do_GET = _forward
        do_POST = _forward

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return ForwardingHandler


class _LoopbackServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_with_proxy(
    upstream_url: str,
    command: Sequence[str],
    *,
    timeout: int = DEFAULT_UPSTREAM_TIMEOUT_SECONDS,
) -> int:
    """Run a Lemonade CLI command through a temporary loopback bridge."""

    if not command:
        raise ValueError("A Lemonade CLI command is required")
    upstream = _validated_upstream(upstream_url)
    server = _LoopbackServer(("127.0.0.1", 0), _handler(upstream, timeout))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    child_command = [str(command[0]), "--host", f"http://{host}:{port}"]
    child_command.extend(str(argument) for argument in command[1:])
    try:
        return subprocess.run(child_command, check=False).returncode
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument(
        "--upstream-timeout", type=int, default=DEFAULT_UPSTREAM_TIMEOUT_SECONDS
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    try:
        return run_with_proxy(
            args.upstream,
            command,
            timeout=max(1, args.upstream_timeout),
        )
    except (OSError, ValueError) as exc:
        print(f"Lemonade Bench proxy error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
