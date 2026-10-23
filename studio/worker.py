"""Long-running installer/generator worker.

Called only as a subprocess from studio.dashboard. Its stdout/stderr are
persisted in /content/AnimeColabStudio/runtime.log.
"""
from __future__ import annotations

import os
import secrets
import sys
import threading
import time
import traceback
from pathlib import Path

# When executed as a file, make package imports possible.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Write the very first log line before importing any optional AI/Colab code.
print("[BOOT] Proceso Python iniciado; importando componentes mínimos...", flush=True)
from studio.runtime import CONFIG, LOG, STATUS, read_json, save_pid, stage, RUNTIME

launcher = None
CURRENT = {"label": "Iniciando", "busy": True}
SHUTDOWN = threading.Event()


def heartbeat():
    started = time.monotonic()
    while not SHUTDOWN.wait(12):
        if not CURRENT["busy"]:
            return
        elapsed = int(time.monotonic() - started)
        print(f"[HEARTBEAT] Activo | {CURRENT['label']} | {elapsed}s transcurridos.", flush=True)


def progress(step, msg):
    CURRENT["label"] = msg
    stage(step, msg)


def validate_config(cfg):
    if cfg.get("model") not in launcher.MODELS:
        raise ValueError("Modelo desconocido.")
    if cfg.get("ui") not in ("ComfyUI", "AUTOMATIC1111"):
        raise ValueError("Interfaz desconocida.")
    if cfg.get("tunnel") not in ("Cloudflare", "ngrok", "Local only"):
        raise ValueError("Túnel desconocido.")


def start():
    global launcher
    progress("bootstrap", "Worker activo. Cargando componentes ligeros...")
    from studio import launcher as backend
    launcher = backend
    progress("bootstrap", "Componentes cargados; leyendo preferencias...")
    cfg = read_json(CONFIG)
    validate_config(cfg)
    output = RUNTIME
    output.mkdir(parents=True, exist_ok=True)
    progress("gpu", "Comprobando GPU NVIDIA...")
    launcher.check_gpu()
    progress("installing", f"Preparando {cfg['ui']} y sus dependencias. Esto puede generar bastante salida en el registro.")
    app = launcher.install_ui(cfg["ui"])
    repo, filename, family = launcher.MODELS[cfg["model"]]
    store = (launcher.DRIVE_ROOT if cfg.get("cache_models", True) else launcher.LOCAL_ROOT)
    checkpoint_dir = store / "checkpoints"
    lora_dir = store / "loras"
    token = launcher.secret("HF_TOKEN") or None

    progress("model", f"Descargando/verificando {filename}. Un SDXL completo suele ocupar unos 7 GB.")
    checkpoint = launcher.download_hf(repo, filename, checkpoint_dir, token)
    if checkpoint.stat().st_size < 1_000_000_000:
        raise RuntimeError("Checkpoint incompleto: descarga demasiado pequeña.")
    launcher.safe_link(checkpoint, launcher.model_dir(app, cfg["ui"]) / filename)

    selected = cfg.get("loras", [])
    custom = [x.strip() for x in cfg.get("custom_loras", "").splitlines() if x.strip() and not x.strip().startswith("#")]
    installed = []
    total = len(selected) + len(custom)
    progress("loras", f"Preparando {total} LoRA(s) seleccionados...")
    for index, name in enumerate(selected, start=1):
        if name not in launcher.LORAS:
            raise ValueError(f"LoRA no reconocido: {name}")
        lora_repo, lora_filename, compatible, _ = launcher.LORAS[name]
        if family not in compatible:
            raise ValueError(f"LoRA '{name}' incompatible con la familia del modelo '{family}'.")
        progress("loras", f"LoRA {index}/{total}: {name}")
        item = launcher.download_hf(lora_repo, lora_filename, lora_dir, token)
        launcher.safe_link(item, launcher.lora_dir(app, cfg["ui"]) / item.name)
        installed.append(item.name)
    for index, line in enumerate(custom, start=len(selected)+1):
        progress("loras", f"LoRA personalizado {index}/{total}: descargando...")
        item = launcher.download_custom_lora(line, lora_dir)
        launcher.safe_link(item, launcher.lora_dir(app, cfg["ui"]) / item.name)
        installed.append(item.name)
    launcher.STATE["installed_loras"] = installed
    if cfg["ui"] == "AUTOMATIC1111" and cfg.get("adetailer", False):
        progress("extras", "Instalando ADetailer...")
        extension = app / "extensions" / "adetailer"
        if not (extension / "scripts").exists():
            if extension.exists():
                launcher.shutil.rmtree(extension)
            launcher.run(["git", "clone", "--depth", "1", "https://github.com/Bing-su/adetailer.git", str(extension)])

    destination = (launcher.DRIVE_ROOT if cfg.get("save_to_drive", True) else launcher.LOCAL_ROOT) / "outputs" / cfg["model"].replace(" ", "_")
    destination.mkdir(parents=True, exist_ok=True)
    progress("server", f"Iniciando {cfg['ui']} (modelos cargándose en la GPU)...")
    port, app_log = launcher.launch_ui(app, cfg, destination)
    save_pid("server", launcher.STATE["ui_process"].pid)
    launcher.wait_for_http(port, launcher.STATE["ui_process"], app_log, timeout=900, ui_name=cfg["ui"])
    password = ""
    source = ""
    verified = True
    if cfg["tunnel"] != "Local only":
        progress("proxy", "Preparando acceso con contraseña de sesión (sin código de correo)...")
        source = "WEBUI_PASSWORD" if launcher.secret("WEBUI_PASSWORD") else "generada"
        password = launcher.secret("WEBUI_PASSWORD") or secrets.token_urlsafe(16)
        proxy_port, proxy_log = launcher.launch_access_proxy(port, password)
        progress("tunnel", f"Abriendo conexión {cfg['tunnel']} sin autenticación por correo...")
        url = launcher.launch_tunnel(cfg["tunnel"], proxy_port)
        progress("web_check", "Probando Web UI y sesión autenticada a través de la URL pública...")
        verified = launcher.verify_tunnel_public_url(url, password, ui_name=cfg["ui"])
    else:
        url = f"http://127.0.0.1:{port} (solo desde Colab)"
    final_stage = "ready" if verified else "degraded"
    final_msg = ("Web UI y acceso externo autenticado comprobados."
                 if verified else "Servidor iniciado, pero no se confirmó el acceso externo; revisa la URL y los registros.")
    stage(final_stage, final_msg, url=url, output_dir=str(destination),
          model=cfg["model"], ui=cfg["ui"], loras=installed,
          access_password=password if source == "generada" else "",
          password_source=source, tunnel_verified=verified)
    print(f"URL: {url}", flush=True)
    if source == "generada":
        print("Contraseña de sesión creada; se muestra en el panel de Colab y en la celda 3.", flush=True)
    elif source == "WEBUI_PASSWORD":
        print("Acceso con la contraseña guardada en el secreto WEBUI_PASSWORD de Colab.", flush=True)
    print("Verificación externa: " + ("CORRECTA" if verified else "NO CONFIRMADA"), flush=True)
    print(f"Imágenes PNG: {destination}", flush=True)
    print("LoRAs instalados: " + (", ".join(installed) or "(ninguno)"), flush=True)
    CURRENT["busy"] = False


if __name__ == "__main__":
    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        start()
    except Exception as exc:
        print("\n" + traceback.format_exc(), flush=True)
        stage("error", f"{type(exc).__name__}: {exc}. Consulta el registro completo.")
        if launcher is not None:
            for key in ("tunnel_process", "proxy_process", "ui_process"):
                proc = launcher.STATE.get(key)
                if proc and proc.poll() is None:
                    proc.terminate()
    finally:
        CURRENT["busy"] = False
        SHUTDOWN.set()
