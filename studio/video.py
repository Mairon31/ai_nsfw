"""Video backends for Anime Colab Studio (GPU-dependent runtime).

Wan2.2 TI2V-5B: ComfyUI native workflow and official Hugging Face weights.
WanGP: curated model discovery via WanGP's own UI (downloads on demand).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

# UI suggestions, NOT made-up weights URLs. WanGP downloads the selected model
# in its own interface, where availability and resource usage are visible.
VIDEO_PRESETS = {
    "Wan 2.2 TI2V 5B · ComfyUI": {
        "backend": "ComfyUI", "recommended_vram": 8,
        "description": "Texto + imagen a video, plantilla oficial (T4 con offload; lento).",
    },
    "Wan 2.2 T2V / I2V · WanGP": {
        "backend": "WanGP", "recommended_vram": 16,
        "description": "Variantes Wan 2.2 (14B son lentas y requieren offload).",
    },
    "Wan 2.1 1.3B · WanGP": {
        "backend": "WanGP", "recommended_vram": 8,
        "description": "Punto de partida más ligero para texto a video.",
    },
    "LTX-2.x · WanGP": {
        "backend": "WanGP", "recommended_vram": 16,
        "description": "Familia LTX disponible en el selector interno de WanGP.",
    },
    "HunyuanVideo · WanGP": {
        "backend": "WanGP", "recommended_vram": 16,
        "description": "Familia Hunyuan optimizada; perfil de baja VRAM.",
    },
}
WAN_GP_REPO = "https://github.com/DeepBeepMeep/Wan2GP.git"
WAN_PYTHON = Path("/content/AnimeColabStudio/wan-venv")
WORKFLOW_NAME = "AnimeStudio-Wan22-TI2V-5B.json"
WORKFLOW_URL = (
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/"
    "templates/video_wan2_2_5B_ti2v.json"
)
WAN22_REPO = "Comfy-Org/Wan-2.2"
WAN22_FILES = (
    ("split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
     "diffusion_models", 5_000_000_000),
    ("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     "text_encoders", 2_000_000_000),
    ("split_files/vae/wan2.2_vae.safetensors",
     "vae", 700_000_000),
)


def info(preset: str) -> dict:
    if preset not in VIDEO_PRESETS:
        raise ValueError("Preset de video desconocido.")
    return VIDEO_PRESETS[preset]


def ensure_wangp_python() -> Path:
    """Separate Python 3.11/torch environment, never mutate Colab image runtime."""
    from studio import launcher
    py = WAN_PYTHON / "bin" / "python"
    if not py.exists():
        uv = shutil.which("uv")
        if not uv:
            launcher.run([sys.executable, "-m", "pip", "install", "-q", "uv"])
            uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("No se pudo preparar uv para WanGP.")
        launcher.run([uv, "venv", "--seed", "--python", "3.11",
                      "--managed-python", str(WAN_PYTHON)])
    result = subprocess.run([str(py), "-c",
                             "import sys; assert sys.version_info[:2] == (3, 11)"],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("El entorno Python 3.11 de WanGP no está listo: " + result.stderr)
    return py


def connect_storage(local_path: Path, persistent_path: Path) -> None:
    """Connect WanGP media/model directory to Drive without discarding files."""
    persistent_path.mkdir(parents=True, exist_ok=True)
    if local_path.is_symlink():
        if local_path.resolve() == persistent_path.resolve():
            return
        local_path.unlink()
    elif local_path.exists():
        if not local_path.is_dir():
            raise RuntimeError(f"Se esperaba una carpeta para WanGP: {local_path}")
        # Preserve files of a previous run before installing a symlink.
        for child in list(local_path.iterdir()):
            destination = persistent_path / child.name
            if destination.exists():
                raise FileExistsError(f"Colisión de archivos; no se borrará: {destination}")
            shutil.move(str(child), str(destination))
        local_path.rmdir()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.symlink_to(persistent_path, target_is_directory=True)


def prepare_wangp(output_root: Path, *, cache_models: bool = True) -> tuple[Path, Path]:
    from studio import launcher
    root = launcher.ROOT / "WanGP"
    py = ensure_wangp_python()
    if not (root / "wgp.py").is_file():
        if root.exists():
            shutil.rmtree(root)
        print("WanGP: descargando frontend, motores de video y configuraciones...", flush=True)
        launcher.run(["git", "clone", "--depth", "1", WAN_GP_REPO, str(root)])
    marker = WAN_PYTHON / ".anime-studio-deps-ready"
    if not marker.exists():
        print("WanGP: instalando PyTorch CUDA 13.0 en Python 3.11 aislado.", flush=True)
        launcher.run([str(py), "-m", "pip", "install", "torch==2.10.0",
                      "torchvision==0.25.0", "torchaudio==2.10.0",
                      "--index-url", "https://download.pytorch.org/whl/cu130"])
        print("WanGP: instalando sus dependencias (primera vez puede ser largo).", flush=True)
        launcher.run([str(py), "-m", "pip", "install", "-r", str(root / "requirements.txt")])
        verify = subprocess.run([str(py), "-c", "import torch; assert torch.cuda.is_available()"],
                                capture_output=True, text=True)
        if verify.returncode:
            raise RuntimeError("WanGP no detecta CUDA con PyTorch: " + verify.stderr[-1200:])
        marker.write_text("ready")
    output_root.mkdir(parents=True, exist_ok=True)
    # Native WanGP uses ckpts/ for downloaded model weights. Keep large
    # downloads and LoRAs in Drive across ephemeral Colab VMs when requested.
    if cache_models:
        base = launcher.DRIVE_ROOT / "video-cache" / "wangp"
        for name in ("ckpts", "loras", "loras_i2v"):
            connect_storage(root / name, base / name)
    for name in ("outputs", "gradio_outputs"):
        connect_storage(root / name, output_root / name)
    return root, py


def prepare_wan22_comfy(app: Path, store: Path, token: str | None = None) -> Path:
    """Download three official files and the native UI workflow, only when chosen."""
    from studio import launcher
    for filename, subfolder, minimum in WAN22_FILES:
        print(f"Wan 2.2: verificando {Path(filename).name}...", flush=True)
        downloaded = launcher.download_hf(WAN22_REPO, filename, store / subfolder, token)
        if downloaded.stat().st_size < minimum:
            raise RuntimeError(f"Modelo de video incompleto: {filename}")
        launcher.safe_link(downloaded, app / "models" / subfolder / Path(filename).name)

    workflow_dir = app / "user" / "default" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    target = workflow_dir / WORKFLOW_NAME
    response = requests.get(WORKFLOW_URL, timeout=45)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise RuntimeError("La plantilla oficial de Wan 2.2 no tiene formato ComfyUI válido.")
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return target


def launch_video_ui(preset: str, app: Path, output_root: Path, python_exe: Path | None = None) -> tuple[int, Path]:
    from studio import launcher
    backend = info(preset)["backend"]
    output_root.mkdir(parents=True, exist_ok=True)
    if backend == "ComfyUI":
        port = 8188
        cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port",
               str(port), "--output-directory", str(output_root)]
        env = os.environ.copy()
    else:
        port = 7862
        if python_exe is None:
            raise ValueError("Falta Python aislado para WanGP.")
        workspace = output_root / "workspaces"
        conf = output_root / "configuration"
        workspace.mkdir(parents=True, exist_ok=True)
        conf.mkdir(parents=True, exist_ok=True)
        cmd = [str(python_exe), "-u", "wgp.py", "--server-port", str(port),
               "--server-name", "127.0.0.1", "--profile", "4",
               "--attention", "sdpa", "--config", str(conf),
               "--workspaces-dir", str(workspace)]
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(WAN_PYTHON)
        env["PATH"] = str(WAN_PYTHON / "bin") + os.pathsep + env.get("PATH", "")
        env["PYTHONNOUSERSITE"] = "1"
    env["MPLBACKEND"] = "Agg"
    logfile = launcher.LOCAL_ROOT / ("comfyui-server.log" if backend == "ComfyUI" else "wangp-server.log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("ab") as out:
        proc = subprocess.Popen(cmd, cwd=str(app), env=env, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
    launcher.STATE["ui_process"] = proc
    from studio.runtime import save_pid
    save_pid("server", proc.pid)
    return port, logfile
