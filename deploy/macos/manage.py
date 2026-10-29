#!/usr/bin/env python3
"""Manage the local AUREL-Mem launchd jobs without installing login items.

Only ``install-login`` and ``uninstall-login`` touch ~/Library/LaunchAgents.
No command reads the service environment file or prints its contents.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE = ROOT
RUNTIME = ROOT / ".runtime" / "macos"
PYTHON = ROOT / ".venv" / "bin" / "python"
ENV_FILE = ROOT / ".env"
CLOUDFLARED = RUNTIME / "bin" / "cloudflared"
LOGS = RUNTIME / "logs"
LAUNCHCTL = "/bin/launchctl"
LABELS = ("org.aurel-mem.local", "org.aurel-mem.tunnel")
PLISTS = {label: RUNTIME / f"{label}.plist" for label in LABELS}
DOMAIN = f"gui/{os.getuid()}"


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


def expected_plists() -> dict[str, dict]:
    environment = {
        "ENV_FILE": str(ENV_FILE),
        "PYTHONPATH": str(CODE),
        "PYTHONUNBUFFERED": "1",
    }
    common = {
        "WorkingDirectory": str(CODE),
        "EnvironmentVariables": environment,
        "KeepAlive": True,
        "RunAtLoad": True,
        "ThrottleInterval": 10,
        "Umask": 0o077,
    }
    service = {
        **common,
        "Label": LABELS[0],
        "ProgramArguments": [
            "/usr/bin/caffeinate", "-i", str(PYTHON), "-m", "uvicorn",
            "fusion_memory.app:create_app", "--factory", "--host", "127.0.0.1",
            "--port", "18080", "--workers", "1", "--no-access-log",
        ],
        "StandardOutPath": str(LOGS / "service.stdout.log"),
        "StandardErrorPath": str(LOGS / "service.stderr.log"),
    }
    tunnel = {
        **common,
        "Label": LABELS[1],
        "ProgramArguments": [str(PYTHON), str(HERE / "tunnel_runner.py")],
        "StandardOutPath": str(LOGS / "tunnel-runner.stdout.log"),
        "StandardErrorPath": str(LOGS / "tunnel-runner.stderr.log"),
    }
    return {LABELS[0]: service, LABELS[1]: tunnel}


def prepare_plists() -> None:
    required = (PYTHON, ENV_FILE, CLOUDFLARED, HERE / "tunnel_runner.py")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing deployment files: " + ", ".join(missing))
    if not CODE.is_dir():
        raise RuntimeError(f"Application directory is missing: {CODE}")
    if not os.access(PYTHON, os.X_OK) or not os.access(CLOUDFLARED, os.X_OK):
        raise RuntimeError("The Python interpreter and cloudflared must be executable")
    RUNTIME.mkdir(parents=True, exist_ok=True, mode=0o700)
    LOGS.mkdir(parents=True, exist_ok=True, mode=0o700)
    RUNTIME.chmod(0o700)
    LOGS.chmod(0o700)
    expected = expected_plists()
    for label in LABELS:
        atomic_private_write(PLISTS[label], plistlib.dumps(expected[label], sort_keys=True))
    for label in LABELS:
        with PLISTS[label].open("rb") as stream:
            if plistlib.load(stream) != expected[label]:
                raise RuntimeError(f"Generated launchd configuration did not validate: {label}")


def launchctl(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([LAUNCHCTL, *arguments], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=20, check=False)


def require_gui_domain() -> None:
    if launchctl("print", DOMAIN).returncode:
        raise RuntimeError(
            f"No accessible {DOMAIN} launchd session. Run this command from this user's logged-in macOS desktop."
        )


def job_state(label: str) -> dict | None:
    result = launchctl("print", f"{DOMAIN}/{label}")
    if result.returncode:
        return None
    state = re.search(r"^\s*state\s*=\s*([^\n]+)", result.stdout, re.MULTILINE)
    pid = re.search(r"^\s*pid\s*=\s*(\d+)", result.stdout, re.MULTILINE)
    last_exit = re.search(r"^\s*last exit code\s*=\s*([^\n]+)", result.stdout, re.MULTILINE)
    return {
        "state": state.group(1).strip() if state else "loaded",
        "pid": pid.group(1) if pid else "-",
        "last_exit": last_exit.group(1).strip() if last_exit else None,
    }


def checked_launchctl(*arguments: str) -> None:
    result = launchctl(*arguments)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"launchctl {' '.join(arguments)} failed: {detail or result.returncode}")


def start() -> None:
    require_gui_domain()
    prepare_plists()
    for label in LABELS:
        state = job_state(label)
        if state is not None:
            print(f"{label}: already loaded, state={state['state']}, pid={state['pid']}")
        else:
            checked_launchctl("bootstrap", DOMAIN, str(PLISTS[label]))
            print(f"{label}: submitted to {DOMAIN}")


def stop() -> None:
    require_gui_domain()
    errors = []
    for label in reversed(LABELS):
        if job_state(label) is None:
            print(f"{label}: not loaded")
            continue
        try:
            checked_launchctl("bootout", f"{DOMAIN}/{label}")
            deadline = time.monotonic() + 10
            while job_state(label) is not None and time.monotonic() < deadline:
                time.sleep(0.2)
            if job_state(label) is not None:
                raise RuntimeError(f"{label}: still present after bootout; inspect launchctl before restarting")
            print(f"{label}: stopped")
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("; ".join(errors))


def status() -> None:
    states = {}
    for label in LABELS:
        state = job_state(label)
        states[label] = state
        if state is None:
            print(f"{label}: not loaded")
        else:
            suffix = f", last_exit={state['last_exit']}" if state["last_exit"] is not None else ""
            print(f"{label}: state={state['state']}, pid={state['pid']}{suffix}")
    endpoints_file = RUNTIME / "current-endpoints.json"
    if not endpoints_file.is_file():
        print("Public endpoints: not yet published; the tunnel may still be connecting.")
        return
    try:
        endpoints = json.loads(endpoints_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("Public endpoints: generated endpoint file is unavailable or invalid.")
        return
    running = all(state and state["state"] == "running" for state in states.values())
    print("Current public endpoint URLs (authentication key is stored separately):" if running
          else "Last published endpoint URLs (jobs are not both running; availability is unconfirmed):")
    for name in ("base_url", "add_url", "search_url", "health_url", "auth_scheme", "generated_at"):
        value = endpoints.get(name)
        if isinstance(value, str):
            print(f"{name}: {value}")


def login_paths() -> dict[str, Path]:
    folder = Path.home() / "Library" / "LaunchAgents"
    return {label: folder / f"{label}.plist" for label in LABELS}


def inspect_login_links() -> dict[str, Path]:
    links = login_paths()
    for label, link in links.items():
        if not os.path.lexists(link):
            continue
        if not link.is_symlink() or os.readlink(link) != str(PLISTS[label]):
            raise RuntimeError(f"Refusing to overwrite or remove an unrelated login item: {link}")
    return links


def install_login() -> None:
    # Check both destinations before touching either, so conflicts never cause
    # a partially installed set of login links.
    links = inspect_login_links()
    prepare_plists()
    next(iter(links.values())).parent.mkdir(parents=True, exist_ok=True)
    for label, link in links.items():
        if os.path.lexists(link):
            print(f"{label}: expected login link already installed")
            continue
        link.symlink_to(PLISTS[label])
        print(f"{label}: installed login link")
    print("Login links installed. This command does not start jobs; use start for the current session.")


def uninstall_login() -> None:
    links = inspect_login_links()
    for label, link in links.items():
        if os.path.lexists(link):
            link.unlink()
            print(f"{label}: removed expected login link")
        else:
            print(f"{label}: login link not installed")
    print("Login links removed. Running jobs are unchanged; use stop to stop them.")


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status", "restart", "install-login", "uninstall-login"))
    command = parser.parse_args().command
    if sys.platform != "darwin":
        parser.error("These launchd commands require macOS")
    try:
        if command == "restart":
            stop()
            start()
        elif command == "install-login":
            install_login()
        elif command == "uninstall-login":
            uninstall_login()
        else:
            {"start": start, "stop": stop, "status": status}[command]()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
