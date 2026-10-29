#!/usr/bin/env python3
"""Run the local HTTPS tunnel and publish its current endpoint URLs.

The generated Add/Search URLs never contain an authentication key. Clients send
the key separately using Authorization: Token (or Bearer). Endpoint files retain
mode 0600, and credentials are redacted from the tunnel and runner logs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = ROOT / ".runtime" / "macos"
ENV_FILE = ROOT / ".env"
CLOUDFLARED = RUNTIME / "bin" / "cloudflared"
LOGS = RUNTIME / "logs"
PUBLIC_URL = re.compile(r"https://[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.trycloudflare\.com\b", re.IGNORECASE)
ENDPOINT_FILES = (RUNTIME / "current-endpoints.json", RUNTIME / "current-endpoints.txt")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_service_token() -> str:
    token = None
    with ENV_FILE.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = re.match(r"^\s*(?:export\s+)?MEMORY_API_TOKEN\s*=\s*(.*)$", line)
            if not match:
                continue
            try:
                pieces = shlex.split(match.group(1), comments=True, posix=True)
            except ValueError:
                raise RuntimeError("MEMORY_API_TOKEN has invalid environment-file quoting") from None
            if len(pieces) != 1:
                raise RuntimeError("MEMORY_API_TOKEN must be one nonempty value")
            token = pieces[0]
    if not token or not token.isascii() or any(character.isspace() for character in token):
        raise RuntimeError("A nonempty ASCII MEMORY_API_TOKEN without whitespace is required")
    return token


def atomic_private_write(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_endpoints(base_url: str) -> None:
    base_url = base_url.rstrip("/")
    endpoint = {
        "base_url": base_url,
        "add_url": f"{base_url}/add",
        "search_url": f"{base_url}/search",
        "health_url": f"{base_url}/healthz",
        "auth_scheme": "Authorization: Token",
        "generated_at": utc_now(),
    }
    atomic_private_write(ENDPOINT_FILES[0], (json.dumps(endpoint, ensure_ascii=False, indent=2) + "\n").encode())
    lines = ["Send the key separately using Authorization: Token <key> (Bearer also supported).",
             "The key is stored in private service.env; this endpoint file contains no authentication key."]
    lines.extend(f"{key}: {value}" for key, value in endpoint.items())
    atomic_private_write(ENDPOINT_FILES[1], ("\n".join(lines) + "\n").encode())


def main() -> int:
    os.umask(0o077)
    child: subprocess.Popen | None = None
    stop_signal: int | None = None
    stop_deadline: float | None = None
    token = ""
    encoded_token = ""

    def stop(signum, _frame):
        nonlocal stop_signal, stop_deadline
        if stop_signal is None:
            stop_signal = signum
            stop_deadline = time.monotonic() + 10
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        token = read_service_token()
        encoded_token = quote(token, safe="")
        if not CLOUDFLARED.is_file() or not os.access(CLOUDFLARED, os.X_OK):
            raise RuntimeError("cloudflared is missing or not executable")
        LOGS.mkdir(parents=True, exist_ok=True, mode=0o700)
        LOGS.chmod(0o700)
        log_path = LOGS / "cloudflared.log"
        # Previous quick-tunnel hostnames are no longer current when a new
        # process starts; only these two generated artifacts are removed.
        for endpoint_file in ENDPOINT_FILES:
            endpoint_file.unlink(missing_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            os.fchmod(log.fileno(), 0o600)
            log.write(f"{utc_now()} Starting cloudflared for local AUREL-Mem.\n")
            child = subprocess.Popen(
                [str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--protocol", "http2",
                 "--url", "http://127.0.0.1:18080"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                bufsize=0,
            )
            if stop_signal is not None:
                child.send_signal(stop_signal)
            assert child.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            pending = b""
            published_url = None
            killed = False

            def handle_line(raw: bytes) -> None:
                nonlocal published_url
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                # A future cloudflared version might log upstream paths. Avoid
                # recording either literal or URL-encoded credentials.
                safe_line = line.replace(token, "[REDACTED]").replace(encoded_token, "[REDACTED]")
                log.write(safe_line + "\n")
                match = PUBLIC_URL.search(line)
                if match and stop_signal is None and match.group(0) != published_url:
                    published_url = match.group(0)
                    publish_endpoints(published_url)
                    log.write(f"{utc_now()} Public endpoint files updated (private URLs omitted).\n")

            try:
                while selector.get_map():
                    if stop_deadline is not None and time.monotonic() >= stop_deadline and child.poll() is None:
                        if not killed:
                            child.kill()
                            killed = True
                            log.write(f"{utc_now()} cloudflared exceeded shutdown grace period; sent SIGKILL.\n")
                    for key, _events in selector.select(timeout=0.5):
                        block = os.read(key.fileobj.fileno(), 65536)
                        if not block:
                            selector.unregister(key.fileobj)
                            continue
                        pending += block
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            handle_line(line)
                        # Bound buffering if a process writes a malformed very
                        # long line. Normal cloudflared messages are much shorter.
                        if len(pending) > 131072:
                            handle_line(pending)
                            pending = b""
                if pending:
                    handle_line(pending)
                returncode = child.wait(timeout=10)
                log.write(f"{utc_now()} cloudflared exited with code {returncode}.\n")
            finally:
                selector.close()
                child.stdout.close()
        if stop_signal is not None:
            return 0
        return returncode if returncode > 0 else (1 if returncode < 0 else 0)
    except Exception as exc:
        # Do not serialize arbitrary exception messages or local environment
        # contents; errors must not disclose access credentials.
        print(f"tunnel_runner failed ({type(exc).__name__}); inspect deployment paths and private tunnel logs.",
              file=sys.stderr)
        return 1
    finally:
        if child is not None:
            if child.poll() is None:
                try:
                    child.terminate()
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
                except ProcessLookupError:
                    child.wait(timeout=10)
            else:
                child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
