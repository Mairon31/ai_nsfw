"""Process supervision and status for Colab Studio.

Persistent disk logs are used instead of relying on ipywidgets updates from a
background notebook thread. All heavy work takes place in a child process.
"""
from __future__ import annotations

import html
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUNTIME = Path("/content/AnimeColabStudio")
REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS = RUNTIME / "runtime-status.json"
PIDS = RUNTIME / "runtime-pids.json"
LOG = RUNTIME / "runtime.log"
CONFIG = RUNTIME / "runtime-config.json"

STAGES = {
    "preparing": 3,
    "gpu": 7,
    "installing": 18,
    "model": 34,
    "loras": 56,
    "extras": 68,
    "server": 78,
    "tunnel": 92,
    "ready": 100,
    "error": 0,
    "stopped": 0,
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def stage(name: str, message: str, **details) -> None:
    previous = read_json(STATUS)
    value = {
        "stage": name,
        "percent": STAGES.get(name, 0),
        "message": message,
        "updated_at": utc_now(),
        "started_at": previous.get("started_at") or utc_now(),
        **details,
    }
    atomic_json(STATUS, value)
    print(f"[{value['updated_at']}] [{name.upper()}] {message}", flush=True)


def pid_alive(pid: int, *, markers=()) -> bool:
    try:
        if pid <= 1:
            return False
        # Linux Colab runtime; avoid terminating an unrelated reused PID.
        data = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
        return any(marker in data for marker in markers) if markers else bool(data)
    except (OSError, ValueError):
        return False


def save_pid(name: str, pid: int) -> None:
    data = read_json(PIDS)
    data[name] = pid
    atomic_json(PIDS, data)


def process_active() -> bool:
    pids = read_json(PIDS)
    return any(pid_alive(int(pids.get(k, 0)), markers=v) for k, v in (
        ("worker", ("studio/worker.py", "studio.worker")),
        ("server", ("ComfyUI/main.py", "main.py", "launch.py", "webui.py")),
        ("tunnel", ("cloudflared", "ngrok")),
    ))


def stop_session() -> None:
    """Stop active installation, UI, and tunnel, including subprocess groups."""
    pids = read_json(PIDS)
    targets = [
        ("tunnel", ("cloudflared", "ngrok")),
        ("server", ("ComfyUI/main.py", "main.py", "launch.py", "webui.py")),
        ("worker", ("studio/worker.py", "studio.worker")),
    ]
    # SIGTERM process groups so child git/pip installs are also interrupted.
    for kind, patterns in targets:
        pid = pids.get(kind)
        if not isinstance(pid, int) or not pid_alive(pid, markers=patterns):
            continue
        try:
            pgid = os.getpgid(pid)
            if pgid == os.getpgrp():
                os.kill(pid, signal.SIGTERM)
            else:
                os.killpg(pgid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    # Allow cleanup, then force-stop survivors, without holding the UI for long.
    for kind, patterns in targets:
        pid = pids.get(kind)
        if not isinstance(pid, int):
            continue
        if pid_alive(pid, markers=patterns):
            try:
                pgid = os.getpgid(pid)
                if pgid == os.getpgrp():
                    os.kill(pid, signal.SIGKILL)
                else:
                    os.killpg(pgid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
    atomic_json(PIDS, {})
    stage("stopped", "Detenido. Puedes pulsar Iniciar para volver a intentarlo.")


def launch_worker(config: dict, secrets: dict[str, str]) -> int:
    if process_active():
        raise RuntimeError("Ya hay una instalación o servidor activo. Pulsa Detener primero.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    atomic_json(PIDS, {})
    atomic_json(CONFIG, config)
    LOG.write_text(f"Anime Colab Studio | sesión iniciada {utc_now()}\n", encoding="utf-8")
    stage("preparing", "Preparando la ejecución...")
    env = os.environ.copy()
    # Temporary child environment only, never written to disk.
    env.update({k: v for k, v in secrets.items() if v})
    with LOG.open("ab", buffering=0) as writer:
        worker = subprocess.Popen(
            [sys.executable, "-u", str(REPO_ROOT / "studio" / "worker.py")],
            stdin=subprocess.DEVNULL, stdout=writer, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT), env=env, start_new_session=True,
        )
    save_pid("worker", worker.pid)
    return worker.pid


def snapshot() -> dict:
    data = read_json(STATUS)
    pids = read_json(PIDS)
    name = data.get("stage", "idle")
    if name == "ready":
        pid = pids.get("server")
        if not isinstance(pid, int) or not pid_alive(pid, markers=("main.py", "launch.py", "webui.py")):
            data.update(stage="error", message="El servidor terminó. Revisa los logs y pulsa Iniciar para reintentarlo.", percent=0)
            atomic_json(STATUS, data)
    elif name not in {"error", "stopped"} and not process_active() and pids:
        data.update(stage="error", message="El proceso terminó inesperadamente. Consulta el registro.", percent=0)
        atomic_json(STATUS, data)
    return data


def _tail_file(path: Path, limit: int = 40000, lines: int = 25) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            blob = stream.read().decode("utf-8", errors="replace")
        return "\n".join(blob.splitlines()[-lines:])
    except OSError:
        return ""


def recent_log(lines: int = 45) -> str:
    """Show installation progress AND logs from the separate UI/tunnel processes."""
    main = _tail_file(LOG, lines=lines)
    extras = []
    for name in ("comfyui-server.log", "automatic1111-server.log", "tunnel.log"):
        location = RUNTIME / name
        if not location.exists():
            continue
        data = _tail_file(location, limit=18000, lines=16)
        if data:
            extras.append(f"\n--- {name} ---\n{data}")
    return (main or "(Sin registros todavía)") + "".join(extras)


def render_log(lines=40) -> str:
    return '<pre style="white-space:pre-wrap;max-height:300px;overflow-y:auto;background:#161a20;color:#e6edf3;padding:12px;border-radius:8px;font:12px/1.45 monospace">' + html.escape(recent_log(lines)) + "</pre>"
