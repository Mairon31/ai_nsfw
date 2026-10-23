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

RUNTIME = Path(os.environ.get("ANIME_STUDIO_RUNTIME_DIR", "/content/AnimeColabStudio"))
REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS = RUNTIME / "runtime-status.json"
PIDS = RUNTIME / "runtime-pids.json"
LOG = RUNTIME / "runtime.log"
CONFIG = RUNTIME / "runtime-config.json"

STAGES = {
    "preparing": 3,
    "bootstrap": 5,
    "gpu": 7,
    "installing": 18,
    "model": 34,
    "loras": 56,
    "extras": 68,
    "server": 78,
    "proxy": 86,
    "tunnel": 92,
    "web_check": 96,
    "degraded": 98,
    "ready": 100,
    "error": 0,
    "stalled": 0,
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
        ("proxy", ("access_proxy.py",)),
        ("tunnel", ("cloudflared", "ngrok")),
    ))


def stop_session() -> None:
    """Stop active installation, UI, and tunnel, including subprocess groups."""
    pids = read_json(PIDS)
    targets = [
        ("tunnel", ("cloudflared", "ngrok")),
        ("proxy", ("access_proxy.py",)),
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
    atomic_json(STATUS, {})
    for old_log in ("comfyui-server.log", "automatic1111-server.log", "access-proxy.log", "tunnel.log"):
        (RUNTIME / old_log).unlink(missing_ok=True)
    LOG.write_text(f"Anime Colab Studio | sesión iniciada {utc_now()}\n", encoding="utf-8")
    stage("preparing", "Preparando la ejecución...")
    env = os.environ.copy()
    env["ANIME_STUDIO_WORKER"] = "1"
    # Never propagate Colab's notebook-only matplotlib backend to detached
    # installers, extensions, or their subprocesses.
    env["MPLBACKEND"] = "Agg"
    # Temporary child environment only, never written to disk.
    env.update({k: v for k, v in secrets.items() if v})
    with LOG.open("ab", buffering=0) as writer:
        worker = subprocess.Popen(
            [sys.executable, "-u", str(REPO_ROOT / "studio" / "worker.py")],
            stdin=subprocess.DEVNULL, stdout=writer, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT), env=env, start_new_session=True,
        )
    save_pid("worker", worker.pid)
    # Let an immediately-failing interpreter leave a readable status/log instead
    # of a dashboard that appears frozen forever.
    time.sleep(0.35)
    if worker.poll() is not None:
        message = f"El instalador terminó inmediatamente (código {worker.returncode}). Consulta runtime.log."
        stage("error", message)
        raise RuntimeError(message)
    return worker.pid


def snapshot() -> dict:
    data = read_json(STATUS)
    pids = read_json(PIDS)
    name = data.get("stage", "idle")
    if name in {"ready", "degraded"}:
        pid = pids.get("server")
        if not isinstance(pid, int) or not pid_alive(pid, markers=("main.py", "launch.py", "webui.py")):
            data.update(stage="error", message="El servidor terminó. Revisa los logs y pulsa Iniciar para reintentarlo.", percent=0)
            atomic_json(STATUS, data)
    elif name == "preparing" and pids and process_active():
        # A fresh worker should write its bootstrap stage almost immediately.
        # Otherwise make the hang visible instead of showing endless 3%.
        from datetime import datetime
        try:
            when = datetime.fromisoformat(data.get("updated_at", ""))
            age = (datetime.now(timezone.utc) - when).total_seconds()
        except (ValueError, TypeError):
            age = 0
        if age > 30:
            data.update(stage="stalled", message="Worker bloqueado al arrancar (>30 s). Pulsa Detener y consulta logs.", percent=0)
            atomic_json(STATUS, data)
    elif name not in {"error", "stopped", "stalled"} and not process_active() and pids:
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
    for name in ("comfyui-server.log", "automatic1111-server.log", "access-proxy.log", "tunnel.log"):
        location = RUNTIME / name
        if not location.exists():
            continue
        data = _tail_file(location, limit=18000, lines=16)
        if data:
            extras.append(f"\n--- {name} ---\n{data}")
    return (main or "(Sin registros todavía)") + "".join(extras)


def render_log(lines=40) -> str:
    return '<pre style="white-space:pre-wrap;max-height:300px;overflow-y:auto;background:#161a20;color:#e6edf3;padding:12px;border-radius:8px;font:12px/1.45 monospace">' + html.escape(recent_log(lines)) + "</pre>"



def monitor_active_session(interval: int = 20, *, max_checks: int | None = None,
                           check_api: bool = True) -> str:
    """Keep the notebook cell running while an actual image server is alive.

    This is not an artificial anti-idle browser action. It reports process/API
    health and exits on stopped/crashed sessions. The platform may still reclaim
    its VM due to resource limits, regardless of cell execution.
    """
    import urllib.error
    import urllib.request

    interval = max(10, min(120, int(interval)))
    status = snapshot()
    stage_name = status.get("stage", "idle")
    if not process_active() or stage_name in {"idle", "stopped", "error", "stalled"}:
        message = "No hay una instalación o servidor activo. Pulsa Instalar / iniciar en la celda 2 primero."
        print(message, flush=True)
        return "not_running"

    print("SUPERVISIÓN ACTIVA. Esta celda permanecerá ejecutándose mientras el servidor esté disponible.", flush=True)
    print("Pulsa el botón de detener ejecución de la celda para salir del monitor.", flush=True)
    print("Colab mantiene sus propios límites de sesión y GPU; este monitor no puede modificarlos.", flush=True)
    if status.get("url"):
        print("Web UI: " + str(status["url"]), flush=True)
    if status.get("output_dir"):
        print("Imágenes: " + str(status["output_dir"]), flush=True)

    checks = 0
    consecutive_errors = 0
    last_stage = None
    while True:
        status = snapshot()
        state = status.get("stage", "idle")
        if state in {"error", "stopped", "stalled"} or not process_active():
            print("SUPERVISIÓN FINALIZADA | " + status.get("message", str(state)), flush=True)
            print(recent_log(12), flush=True)
            return state if state in {"error", "stopped", "stalled"} else "not_running"

        if state not in {"ready", "degraded"}:
            if state != last_stage or checks % max(1, 60 // interval) == 0:
                print(f"[{utc_now()}] Preparando el servidor | etapa={state} | "
                      f"{status.get('message', '')}", flush=True)
            last_stage = state
            checks += 1
            if max_checks is not None and checks >= max_checks:
                return state
            time.sleep(interval)
            continue

        data = read_json(CONFIG)
        mode = data.get("ui", status.get("ui", "ComfyUI"))
        port = 7860 if mode == "AUTOMATIC1111" else 8188
        endpoint = "/sdapi/v1/options" if mode == "AUTOMATIC1111" else "/system_stats"

        api_result = "sin comprobar"
        if check_api:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{endpoint}", timeout=4
                ) as response:
                    api_result = "OK" if response.status == 200 else f"HTTP {response.status}"
            except (OSError, urllib.error.URLError, ValueError) as exc:
                api_result = type(exc).__name__
            if api_result == "OK":
                consecutive_errors = 0
            else:
                consecutive_errors += 1

        if state != last_stage or consecutive_errors == 3 or checks % max(1, 60 // interval) == 0:
            print(f"[{utc_now()}] sesión={state} | servidor API={api_result} | "
                  f"comprobaciones={checks + 1}", flush=True)
            if consecutive_errors == 3:
                print("ATENCIÓN: tres comprobaciones fallaron. "
                      "Si estás generando imágenes, puede ser una respuesta tardía; "
                      "consulta los logs antes de reiniciar.", flush=True)
            last_stage = state

        checks += 1
        if max_checks is not None and checks >= max_checks:
            return state
        time.sleep(interval)



async def monitor_active_session_async(
    interval: int = 20, *, max_checks: int | None = None, check_api: bool = True
) -> str:
    """Monitor real server activity without blocking Jupyter's widget event loop.

    Intended for the notebook monitoring cell. A running cell is useful for
    real work but does not override Colab resource/usage limits.
    """
    import asyncio
    import urllib.request
    import urllib.error

    interval = max(10, min(120, int(interval)))
    current = snapshot()
    if not process_active() or current.get("stage") in {"idle", "stopped", "error", "stalled"}:
        print("No hay instalación ni servidor activo. Pulsa Instalar / iniciar antes de ejecutar esta celda.", flush=True)
        return "not_running"

    print("MONITOR DE SESIÓN ACTIVO | La celda seguirá ejecutándose mientras el trabajo continúe.", flush=True)
    print("Puedes continuar usando la Web UI y los botones de la celda 2.", flush=True)
    print("Detener esta celda cancela solamente el monitor; no apaga el servidor.", flush=True)
    print("Los límites de tiempo, GPU y unidades de Colab siguen vigentes.", flush=True)

    checks, api_failures, last_stage = 0, 0, None
    first_ready_reported = False
    try:
        while True:
            current = snapshot()
            state = current.get("stage", "idle")
            if state in {"error", "stopped", "stalled"} or not process_active():
                print("MONITOR FINALIZADO: " + current.get("message", str(state)), flush=True)
                print(recent_log(15), flush=True)
                return str(state)

            if state not in {"ready", "degraded"}:
                if last_stage != state or checks % max(1, 60 // interval) == 0:
                    print(f"[{utc_now()}] Instalación en curso: {state} | "
                          + str(current.get("message", "")), flush=True)
            else:
                if not first_ready_reported:
                    first_ready_reported = True
                    if current.get("url"):
                        print("Interfaz: " + str(current["url"]), flush=True)
                    if current.get("output_dir"):
                        print("Archivos: " + str(current["output_dir"]), flush=True)
                    print("Contraseña remota: consulta el panel de la celda 2 o el diagnóstico de la celda 3.", flush=True)

                api_result = "no comprobada"
                if check_api:
                    cfg = read_json(CONFIG)
                    ui = cfg.get("ui", current.get("ui", "ComfyUI"))
                    port = 7860 if ui == "AUTOMATIC1111" else 8188
                    endpoint = "/sdapi/v1/options" if ui == "AUTOMATIC1111" else "/system_stats"
                    def probe():
                        try:
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{port}{endpoint}", timeout=4
                            ) as response:
                                return "OK" if response.status == 200 else f"HTTP {response.status}"
                        except (OSError, ValueError) as exc:
                            return type(exc).__name__
                    api_result = await asyncio.to_thread(probe)
                    api_failures = 0 if api_result == "OK" else api_failures + 1
                if last_stage != state or checks % max(1, 60 // interval) == 0 or api_failures == 3:
                    print(f"[{utc_now()}] Web UI: {state} | API: {api_result} | "
                          f"revisiones: {checks + 1}", flush=True)
                    if api_failures == 3:
                        print("API no respondió en tres revisiones. Podría estar ocupada generando; consulta los logs.", flush=True)

            last_stage = state
            checks += 1
            if max_checks is not None and checks >= max_checks:
                return str(state)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        print("Monitor cancelado. La Web UI y el túnel pueden continuar ejecutándose.", flush=True)
        raise


def launch_foreground(config: dict, secrets: dict[str, str]) -> int:
    """Colab fallback: run the worker in an executing cell with streamed output.

    Unlike launching from a widget callback, the cell remains in RUNNING state
    until the worker has initialized the Web UI and tunnel (or failed).
    """
    if process_active():
        stop_session()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    atomic_json(PIDS, {})
    atomic_json(CONFIG, config)
    atomic_json(STATUS, {})
    for old_log in ("comfyui-server.log", "automatic1111-server.log", "access-proxy.log", "tunnel.log"):
        (RUNTIME / old_log).unlink(missing_ok=True)
    LOG.write_text(f"Anime Colab Studio | modo directo {utc_now()}\n", encoding="utf-8")
    stage("preparing", "Modo directo: lanzando proceso supervisado en esta celda...")
    env = os.environ.copy()
    env["ANIME_STUDIO_WORKER"] = "1"
    # Never propagate Colab's notebook-only matplotlib backend to detached
    # installers, extensions, or their subprocesses.
    env["MPLBACKEND"] = "Agg"
    env.update({key: val for key, val in secrets.items() if val})
    command = [sys.executable, "-u", str(REPO_ROOT / "studio" / "worker.py")]
    proc = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, cwd=str(REPO_ROOT),
        env=env, text=True, bufsize=1, start_new_session=True,
    )
    save_pid("worker", proc.pid)
    print(f"Worker PID {proc.pid}. La celda mostrará el progreso en tiempo real.", flush=True)
    try:
        with LOG.open("a", encoding="utf-8", buffering=1) as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                print(line, end="", flush=True)
        exit_code = proc.wait()
        current = read_json(STATUS)
        if exit_code != 0 and current.get("stage") not in {"error", "stopped"}:
            stage("error", f"El worker terminó con código {exit_code}. Consulta los registros.")
        elif exit_code == 0 and current.get("stage") in {"ready", "degraded"}:
            print("Instalación terminada; manteniendo esta celda activa para supervisar el servidor.", flush=True)
            monitor_active_session()
        return exit_code
    except KeyboardInterrupt:
        print("\nInterrupción solicitada. Cancelando worker, servidor y túnel...", flush=True)
        stop_session()
        raise
