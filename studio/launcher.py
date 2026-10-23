"""Interactive Colab launcher. No secrets are written to Git or Drive."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
# Deliberately lazy-load Colab/IPython/Hugging Face modules.
# Importing notebook modules inside a detached worker can block at startup.

VERSION = "1.1.0"
ROOT = Path("/content")
DRIVE_MOUNT = ROOT / "drive"
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive" / "AnimeColabStudio"
LOCAL_ROOT = ROOT / "AnimeColabStudio"
SETTINGS = DRIVE_ROOT / "settings.json"

MODELS = {
    "NoobAI-XL 1.0": ("Laxhar/noobai-XL-1.0", "NoobAI-XL-v1.0.safetensors", "noob"),
    "Pony Diffusion V6 XL": ("6chan/Pony-Diffusion-V6-XL", "ponyDiffusionV6XL_v6StartWithThisOne.safetensors", "pony"),
    "WAI-Illustrious SDXL v17": ("yufusoft/WAI-illustrious-SDXL", "waiIllustriousSDXL_v170.safetensors", "illustrious"),
    "WAI-NSFW-Illustrious v12 (legacy)": ("TheReprinter/WAI-NSFW-illustrious-SDXL", "waiNSFWIllustrious_v12.safetensors", "illustrious"),
    "Animagine XL 4.0 Opt": ("cagliostrolab/animagine-xl-4.0", "animagine-xl-4.0-opt.safetensors", "sdxl"),
    "Illustrious XL 1.0": ("drawhisper/illustrious-xl", "illustrious-xl.safetensors", "illustrious"),
}
LORAS = {
    "Anime Style XL (general SDXL)": ("SUPER321/my-animestyle-lora", "pytorch_lora_weights.safetensors", ("sdxl", "noob"), 0.65),
    "Sensual Face v3 Type 1 (Illustrious)": ("JujoHotaru/lora", "sdxl/sensualface1_xl_v30.safetensors", ("illustrious",), 0.7),
    "Sensual Face v3 Type 2 (Illustrious)": ("JujoHotaru/lora", "sdxl/sensualface2_xl_v30.safetensors", ("illustrious",), 0.7),
    "Sensual Face v2 Type 1 (Pony)": ("JujoHotaru/lora", "sdxl/sensualface1_xl_v20.safetensors", ("pony",), 0.7),
    "Sensual Face v2 Type 2 (Pony)": ("JujoHotaru/lora", "sdxl/sensualface2_xl_v20.safetensors", ("pony",), 0.7),
}
DEFAULTS = {
    "model": "NoobAI-XL 1.0",
    "ui": "ComfyUI",
    "tunnel": "Cloudflare",
    "allowed_email": "",
    "save_to_drive": True,
    "cache_models": True,
    "loras": [],
    "custom_loras": "",
    "adetailer": False,
    "profile": "Balanced",
}
STATE = {"ui_process": None, "tunnel_process": None, "proxy_process": None}

def secret(name: str) -> str:
    """Read secrets from Colab in the notebook, or the temporary worker environment."""
    value = (os.environ.get(name) or "").strip()
    if value:
        return value
    # Detached Python workers cannot use notebook-only Colab user-data RPC.
    # All secrets were already forwarded by the interactive dashboard.
    if os.environ.get("ANIME_STUDIO_WORKER") == "1":
        return ""
    try:
        from google.colab import userdata
        return (userdata.get(name) or "").strip()
    except Exception:
        return ""

def check_gpu() -> None:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("No hay GPU NVIDIA. Colab > Entorno de ejecución > Cambiar tipo > GPU.")
    print(result.stdout.splitlines()[0])

def mount_drive() -> None:
    if not (DRIVE_MOUNT / "MyDrive").exists():
        from google.colab import drive
        drive.mount(str(DRIVE_MOUNT))
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}
    except (OSError, ValueError):
        pass
    return DEFAULTS.copy()

def save_settings(config: dict) -> None:
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    safe_config = {key: config[key] for key in DEFAULTS}
    temp = SETTINGS.with_suffix(".tmp")
    temp.write_text(json.dumps(safe_config, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(SETTINGS)

def run(command, *, cwd=None, env=None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)

A1111_STABLE_DIFFUSION_MIRROR = "https://github.com/w-e-w/stablediffusion.git"
A1111_PYTHON_DIR = LOCAL_ROOT / "a1111-venv"


def ensure_a1111_python() -> Path:
    """Install Python 3.10 privately for legacy AUTOMATIC1111 on Colab 3.13.

    The notebook and ComfyUI retain Colab's own Python interpreter. Never
    replace the system interpreter or install A1111 packages into it.
    """
    interpreter = A1111_PYTHON_DIR / "bin" / "python"
    if interpreter.exists():
        test = subprocess.run([str(interpreter), "-c",
                               "import sys, pip; assert sys.version_info[:2] == (3, 10)"],
                              capture_output=True, text=True)
        if test.returncode == 0:
            print("AUTOMATIC1111: reutilizando Python 3.10 aislado: " + str(interpreter), flush=True)
            return interpreter
        print("AUTOMATIC1111: entorno Python incompleto, reconstruyendo.", flush=True)
        shutil.rmtree(A1111_PYTHON_DIR, ignore_errors=True)

    uv = shutil.which("uv")
    if not uv:
        print("AUTOMATIC1111: instalando gestor uv...", flush=True)
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("No fue posible instalar uv para preparar Python 3.10.")

    print("AUTOMATIC1111: creando Python 3.10 aislado (Colab puede seguir en 3.13)...", flush=True)
    A1111_PYTHON_DIR.parent.mkdir(parents=True, exist_ok=True)
    run([uv, "venv", "--seed", "--python", "3.10", "--managed-python", str(A1111_PYTHON_DIR)])
    test = subprocess.run([str(interpreter), "-c",
                           "import sys, pip; assert sys.version_info[:2] == (3, 10); print(sys.version)"],
                          capture_output=True, text=True)
    if test.returncode != 0:
        raise RuntimeError("Python 3.10 no quedó listo: " + (test.stderr or test.stdout))
    print("AUTOMATIC1111: Python listo: " + test.stdout.strip(), flush=True)
    return interpreter


A1111_CLIP_URL = "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip"
A1111_SETUPTOOLS_VERSION = "69.5.1"


def ensure_a1111_clip(interpreter: Path) -> None:
    """Pre-install legacy OpenAI CLIP without PEP 517 build isolation.

    The pinned CLIP setup.py imports pkg_resources, removed by recent
    setuptools releases. Installing it inside a disposable isolated build
    environment ignores our pinned setuptools and fails on modern Colab.
    This does not install/downgrade torch; A1111 handles its GPU build.
    """
    verify = subprocess.run(
        [str(interpreter), "-c",
         'import importlib.metadata as md; md.version("clip"); import pkg_resources'],
        capture_output=True, text=True,
    )
    if verify.returncode == 0:
        print("AUTOMATIC1111: CLIP ya instalado; omitiendo recompilación.", flush=True)
        return

    print("AUTOMATIC1111: preparando setuptools compatible con CLIP antiguo...", flush=True)
    run([str(interpreter), "-m", "pip", "install",
         "--disable-pip-version-check", "--prefer-binary",
         f"setuptools=={A1111_SETUPTOOLS_VERSION}", "wheel",
         "ftfy", "regex", "tqdm"])

    print("AUTOMATIC1111: instalando CLIP sin build isolation (sin alterar PyTorch)...", flush=True)
    run([str(interpreter), "-m", "pip", "install", "--disable-pip-version-check",
         "--no-build-isolation", "--no-deps", A1111_CLIP_URL])
    result = subprocess.run(
        [str(interpreter), "-c",
         'import importlib.metadata as md; print("clip version:", md.version("clip")); import pkg_resources'],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("CLIP no quedó instalado correctamente: " + (result.stderr or result.stdout))
    print("AUTOMATIC1111: " + result.stdout.strip(), flush=True)


def repair_a1111_stable_diffusion_clone(app: Path) -> None:
    """Remove only a corrupt/incomplete dependency clone, never model caches.

    This repairs Git error 128 from old installations without touching any
    checkpoints, extensions, LoRAs or user data.
    """
    dependency = app / "repositories" / "stable-diffusion-stability-ai"
    if not dependency.exists():
        return
    if not (dependency / ".git").exists():
        print("AUTOMATIC1111: eliminando clon incompleto de Stable Diffusion.", flush=True)
        shutil.rmtree(dependency)
        return
    check = subprocess.run(["git", "-C", str(dependency), "rev-parse", "--verify", "HEAD"],
                           capture_output=True, text=True)
    if check.returncode:
        print("AUTOMATIC1111: reparando repositorio Git incompleto de Stable Diffusion.", flush=True)
        shutil.rmtree(dependency)


def install_ui(ui_name: str) -> Path:
    if ui_name == "ComfyUI":
        path = ROOT / "ComfyUI"
        if not (path / "main.py").exists():
            # A cancelled git clone may leave an incomplete non-empty directory.
            if path.exists():
                shutil.rmtree(path)
            run(["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(path)])
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(path / "requirements.txt")])
    else:
        python310 = ensure_a1111_python()
        ensure_a1111_clip(python310)
        path = ROOT / "stable-diffusion-webui"
        if not (path / "launch.py").exists():
            if path.exists():
                shutil.rmtree(path)
            run(["git", "clone", "--depth", "1", "https://github.com/AUTOMATIC1111/stable-diffusion-webui.git", str(path)])
        repair_a1111_stable_diffusion_clone(path)
    return path

def model_dir(app: Path, ui_name: str) -> Path:
    return app / "models" / ("checkpoints" if ui_name == "ComfyUI" else "Stable-diffusion")

def lora_dir(app: Path, ui_name: str) -> Path:
    return app / "models" / ("loras" if ui_name == "ComfyUI" else "Lora")

def safe_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"No se sobrescribirá: {destination}")
    destination.symlink_to(source)

def download_hf(repo: str, filename: str, folder: Path, token: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / Path(filename).name
    if file.exists() and file.stat().st_size > 1024 * 1024:
        return file
    # Download into the Hugging Face cache and copy only after the file is complete.
    downloaded = Path(hf_hub_download(repo_id=repo, filename=filename, token=token or None))
    temporary = file.with_suffix(file.suffix + ".part")
    shutil.copyfile(downloaded, temporary)
    temporary.replace(file)
    return file

def parse_custom_lora(line: str) -> tuple[str, str]:
    # Supported: hf://owner/repo/path.safetensors or direct Civitai/Hugging Face HTTPS link.
    line = line.strip()
    if line.startswith("hf://"):
        match = re.fullmatch(r"hf://([^/]+)/([^/]+)/(.+\.safetensors)", line, re.I)
        if not match:
            raise ValueError("Formato HF: hf://usuario/repositorio/archivo.safetensors")
        owner, repository, filename = match.groups()
        return "hf", f"{owner}/{repository}|{filename}"
    url = urlparse(line)
    if url.scheme != "https" or url.hostname not in {"huggingface.co", "civitai.com"}:
        raise ValueError("LoRA personalizado: utiliza hf:// o HTTPS de huggingface.co/civitai.com.")
    if url.hostname == "huggingface.co":
        match = re.match(r"^/([^/]+)/([^/]+)/(?:resolve|blob)/([^/]+)/(.+\.safetensors)$", url.path, re.I)
        if not match:
            raise ValueError("URL Hugging Face: debe ser /resolve/main/ruta/archivo.safetensors")
        owner, repo, revision, filename = match.groups()
        return "hf", f"{owner}/{repo}|{unquote(filename)}|{revision}"
    return "url", line

def download_custom_lora(line: str, folder: Path) -> Path:
    from huggingface_hub import hf_hub_download
    kind, value = parse_custom_lora(line)
    if kind == "hf":
        repo, filename, *revision = value.split("|")
        # Use the API for revision-aware downloads.
        folder.mkdir(parents=True, exist_ok=True)
        basename = Path(filename).name
        target = folder / basename
        if target.exists() and target.stat().st_size > 1024:
            return target
        local = Path(hf_hub_download(repo, filename, revision=revision[0] if revision else "main", token=secret("HF_TOKEN") or None))
        part = target.with_suffix(".safetensors.part")
        shutil.copyfile(local, part)
        part.replace(target)
        return target
    folder.mkdir(parents=True, exist_ok=True)
    response = requests.get(value, headers={"Authorization": f"Bearer {secret('CIVITAI_API_TOKEN')}"} if secret("CIVITAI_API_TOKEN") else {}, stream=True, timeout=45)
    response.raise_for_status()
    name = Path(unquote(urlparse(response.url).path)).name
    if not name.lower().endswith(".safetensors"):
        name = "civitai_" + re.search(r"/(\d+)", urlparse(value).path).group(1) + ".safetensors" if re.search(r"/(\d+)", urlparse(value).path) else "custom_lora.safetensors"
    target = folder / name
    part = target.with_suffix(".safetensors.part")
    try:
        with part.open("wb") as fh:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    fh.write(chunk)
        if part.stat().st_size < 1024:
            raise ValueError("Descarga demasiado pequeña; verifica token/URL.")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
        response.close()
    return target

def profile_values(name: str) -> tuple[int, int, int, float]:
    presets = {
        "Balanced": (832, 1216, 28, 6.0),
        "High quality": (1024, 1536, 36, 6.0),
        "Low VRAM": (768, 1024, 22, 5.0),
    }
    return presets.get(name, presets["Balanced"])

def write_a1111_settings(app: Path, outdir: Path, profile: str) -> None:
    width, height, steps, cfg = profile_values(profile)
    config_path = app / "config.json"
    try:
        data = json.loads(config_path.read_text())
    except (OSError, ValueError):
        data = {}
    data.update({
        "samples_save": True,
        "samples_format": "png",
        "save_images_add_number": True,
        "enable_pnginfo": True,
        "outdir_txt2img_samples": str(outdir / "txt2img"),
        "outdir_img2img_samples": str(outdir / "img2img"),
        "outdir_extras_samples": str(outdir / "extras"),
        "outdir_txt2img_grids": str(outdir / "grids"),
        "outdir_img2img_grids": str(outdir / "grids"),
        "grid_save": False,
        "CLIP_stop_at_last_layers": 2,
        "sd_model_checkpoint": None,
        "txt2img/Width/value": width,
        "txt2img/Height/value": height,
        "txt2img/Sampling steps/value": steps,
        "txt2img/CFG Scale/value": cfg,
    })
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def comfy_workflow(filename: str, loras: list[str], profile: str, pony: bool = False) -> dict:
    """Build a native ComfyUI workflow with all selected LoRAs in series."""
    w, h, steps, cfg = profile_values(profile)
    nodes = [{
        "id": 1, "type": "CheckpointLoaderSimple", "pos": [60, 170],
        "size": [330, 100], "flags": {}, "order": 0, "mode": 0,
        "inputs": [],
        "outputs": [{"name": n, "type": n, "links": []} for n in ("MODEL", "CLIP", "VAE")],
        "properties": {"Node name for S&R": "CheckpointLoaderSimple"},
        "widgets_values": [filename]
    }]
    links = []
    def node(nid, typ, x, y, inputs, outputs, values):
        return {
            "id": nid, "type": typ, "pos": [x, y], "size": [315, 110],
            "flags": {}, "order": nid - 1, "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
            "outputs": [{"name": n, "type": t, "links": []} for n, t in outputs],
            "properties": {"Node name for S&R": typ}, "widgets_values": values
        }
    def connect(source, slot, dest, destslot, datatype):
        link_id = len(links) + 1
        links.append([link_id, source, slot, dest, destslot, datatype])
        nodes[source - 1]["outputs"][slot]["links"].append(link_id)
        nodes[dest - 1]["inputs"][destslot]["link"] = link_id

    next_id = 2
    model_src, clip_src = (1, 0), (1, 1)
    if pony:
        nid = next_id
        nodes.append(node(nid, "CLIPSetLastLayer", 410, 370,
                          [("clip", "CLIP")], [("CLIP", "CLIP")], [-2]))
        connect(*clip_src, nid, 0, "CLIP")
        clip_src = (nid, 0)
        next_id += 1

    for index, filename_lora in enumerate(loras):
        nid = next_id
        matched = [v[3] for v in LORAS.values() if Path(v[1]).name == filename_lora]
        weight = matched[0] if matched else 0.75
        nodes.append(node(nid, "LoraLoader", 440 + index * 330, 150,
                          [("model", "MODEL"), ("clip", "CLIP")],
                          [("MODEL", "MODEL"), ("CLIP", "CLIP")],
                          [filename_lora, weight, weight]))
        connect(*model_src, nid, 0, "MODEL")
        connect(*clip_src, nid, 1, "CLIP")
        model_src, clip_src = (nid, 0), (nid, 1)
        next_id += 1

    ids = {}
    regular = [
        ("positive", "CLIPTextEncode", [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")],
         ["masterpiece, best quality, adult woman, solo, detailed anime illustration"], 20),
        ("negative", "CLIPTextEncode", [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")],
         ["low quality, blurry, deformed, watermark, text"], 200),
        ("latent", "EmptyLatentImage", [], [("LATENT", "LATENT")], [w, h, 1], 410),
        ("ksampler", "KSampler", [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
         [("LATENT", "LATENT")], [12345, "randomize", steps, cfg, "dpmpp_2m", "karras", 1.0], 150),
        ("decode", "VAEDecode", [("samples", "LATENT"), ("vae", "VAE")],
         [("IMAGE", "IMAGE")], [], 220),
        ("save", "SaveImage", [("images", "IMAGE")], [], ["AnimeStudio"], 220)
    ]
    for key, typ, ins, outs, values, y in regular:
        nid = next_id
        ids[key] = nid
        next_id += 1
        x = {"positive": 850, "negative": 850, "latent": 850, "ksampler": 1240,
             "decode": 1600, "save": 1950}[key] + len(loras) * 250
        nodes.append(node(nid, typ, x, y, ins, outs, values))
    connect(*clip_src, ids["positive"], 0, "CLIP")
    connect(*clip_src, ids["negative"], 0, "CLIP")
    connect(*model_src, ids["ksampler"], 0, "MODEL")
    connect(ids["positive"], 0, ids["ksampler"], 1, "CONDITIONING")
    connect(ids["negative"], 0, ids["ksampler"], 2, "CONDITIONING")
    connect(ids["latent"], 0, ids["ksampler"], 3, "LATENT")
    connect(ids["ksampler"], 0, ids["decode"], 0, "LATENT")
    connect(1, 2, ids["decode"], 1, "VAE")
    connect(ids["decode"], 0, ids["save"], 0, "IMAGE")
    return {
        "last_node_id": next_id - 1, "last_link_id": len(links),
        "nodes": nodes, "links": links, "groups": [], "config": {},
        "extra": {"ds": {"scale": 0.75, "offset": [100, 90]}}, "version": 0.4
    }

def launch_ui(app: Path, config: dict, output_root: Path) -> tuple[int, Path]:
    ui_name = config["ui"]
    port = 8188 if ui_name == "ComfyUI" else 7860
    logfile = LOCAL_ROOT / f"{ui_name.lower()}-server.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    if ui_name == "ComfyUI":
        args = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(port), "--output-directory", str(output_root)]
        workflow_folder = app / "user" / "default" / "workflows"
        workflow_folder.mkdir(parents=True, exist_ok=True)
        wf = comfy_workflow(MODELS[config["model"]][1], STATE.get("installed_loras", []), config["profile"], pony=MODELS[config["model"]][2] == "pony")
        (workflow_folder / "AnimeStudio.json").write_text(json.dumps(wf, ensure_ascii=False, indent=2))
        (DRIVE_ROOT / "AnimeStudio-workflow.json").write_text(json.dumps(wf, ensure_ascii=False, indent=2))
    else:
        write_a1111_settings(app, output_root, config["profile"])
        ckpt = model_dir(app, ui_name) / MODELS[config["model"]][1]
        interpreter = ensure_a1111_python()
        args = [str(interpreter), "launch.py", "--port", str(port), "--no-download-sd-model", "--ckpt", str(ckpt)]
        # Authentication is enforced at the dedicated tunnel proxy instead.
        # Keeping --gradio-auth here would give API endpoints a second,
        # inconsistent auth mechanism and cause Gradio Unauthorized errors.
        args += ["--api"]
    env = os.environ.copy()
    # Colab exposes its notebook-only matplotlib_inline backend through
    # MPLBACKEND. Our external Web UI runs in a separate Python environment
    # without matplotlib_inline, so always select the headless Agg backend.
    # This changes only the child process, never the Colab notebook kernel.
    env["MPLBACKEND"] = "Agg"
    if ui_name != "ComfyUI":
        env["VENV_DIR"] = "-"  # Already using isolated Python 3.10 venv.
        env["VIRTUAL_ENV"] = str(A1111_PYTHON_DIR)
        env["PATH"] = str(A1111_PYTHON_DIR / "bin") + os.pathsep + env.get("PATH", "")
        env["PYTHONNOUSERSITE"] = "1"
        # AUTOMATIC1111 1.10.1 still references an unavailable upstream repo.
        # Use the mirror explicitly recommended by its maintainers.
        env["STABLE_DIFFUSION_REPO"] = A1111_STABLE_DIFFUSION_MIRROR
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        print("AUTOMATIC1111: origen Stable Diffusion: " + env["STABLE_DIFFUSION_REPO"], flush=True)
        print("AUTOMATIC1111: intérprete aislado: " + str(args[0]), flush=True)
        print("AUTOMATIC1111: Matplotlib backend for headless Web UI: " + env["MPLBACKEND"], flush=True)
    with logfile.open("ab") as stream:
        proc = subprocess.Popen(args, cwd=str(app), env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    STATE["ui_process"] = proc
    try:
        from studio.runtime import save_pid
        save_pid("server", proc.pid)
    except ImportError:
        pass
    return port, logfile

def wait_for_http(port: int, proc: subprocess.Popen, log: Path, timeout: int = 600) -> None:
    for _ in range(timeout // 3):
        if proc.poll() is not None:
            raise RuntimeError(f"Web UI terminó antes de iniciar. Últimas líneas:\n{tail(log)}")
        try:
            response = requests.get(f"http://127.0.0.1:{port}/", timeout=3)
            if response.status_code < 500:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"Web UI no respondió. Registro:\n{tail(log)}")

def tail(path: Path, lines: int = 24) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(log no disponible)"

def launch_tunnel(tunnel: str, port: int, email: str) -> str:
    if tunnel == "Local only":
        return f"http://127.0.0.1:{port} (solo desde Colab)"
    logfile = LOCAL_ROOT / "tunnel.log"
    if tunnel == "Cloudflare":
        executable = ROOT / "cloudflared"
        if executable.exists() and executable.stat().st_size < 5_000_000:
            executable.unlink()
        if not executable.exists():
            res = requests.get("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", timeout=90)
            res.raise_for_status()
            part = executable.with_suffix(".download")
            part.write_bytes(res.content)
            if part.stat().st_size < 5_000_000:
                part.unlink(missing_ok=True)
                raise RuntimeError("Descarga incompleta de cloudflared.")
            part.chmod(0o755)
            part.replace(executable)
        # The --allowed-mail OTP protection can expire mid-session and break
        # Gradio XHR/WebSockets. The protected local proxy handles access.
        command = [str(executable), "--no-autoupdate", "tunnel", "--url", f"http://127.0.0.1:{port}"]
    else:
        token = secret("NGROK_AUTHTOKEN")
        if not token:
            raise ValueError("Configura NGROK_AUTHTOKEN en el panel de Secretos de Colab.")
        executable = ROOT / "ngrok"
        if executable.exists() and executable.stat().st_size < 3_000_000:
            executable.unlink()
        if not executable.exists():
            archive = ROOT / "ngrok.tar.gz"
            response = requests.get("https://bin.ngrok.com/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz", timeout=90)
            response.raise_for_status()
            temporary = archive.with_suffix(".download")
            temporary.write_bytes(response.content)
            if temporary.stat().st_size < 1_000_000:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("Descarga incompleta de ngrok.")
            temporary.replace(archive)
            # Extract into a private staging folder to avoid leaving a partial
            # executable if the user cancels during extraction.
            staging = ROOT / "ngrok-install-partial"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True, exist_ok=True)
            run(["tar", "-xzf", str(archive), "-C", str(staging)])
            candidate = staging / "ngrok"
            if not candidate.exists():
                raise RuntimeError("El archivo descargado no contenía ngrok.")
            candidate.chmod(0o755)
            candidate.replace(executable)
            shutil.rmtree(staging, ignore_errors=True)
        # ngrok reads NGROK_AUTHTOKEN from the temporary worker environment;
        # do not save this credential to a persistent config file.
        os.environ["NGROK_AUTHTOKEN"] = token
        # No email/OAuth policy. Both tunnels point only to the authenticated
        # local proxy, which uses a secure HTTP-only browser cookie.
        command = [str(executable), "http", str(port)]
    with logfile.open("wb") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    STATE["tunnel_process"] = proc
    try:
        from studio.runtime import save_pid
        save_pid("tunnel", proc.pid)
    except ImportError:
        pass
    for _ in range(55):
        if proc.poll() is not None:
            raise RuntimeError(f"No se pudo abrir túnel:\n{tail(logfile)}")
        if tunnel == "Cloudflare":
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", logfile.read_text(errors="replace"))
            if match:
                return match.group(0)
        else:
            try:
                tunnels = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=2).json().get("tunnels", [])
                for item in tunnels:
                    url = item.get("public_url", "")
                    if url.startswith("https://"):
                        return url
            except (requests.RequestException, ValueError):
                pass
        time.sleep(2)
    raise TimeoutError(f"No se encontró URL pública:\n{tail(logfile)}")

ACCESS_PROXY_PORT = 8787

def launch_access_proxy(origin_port: int, password: str) -> tuple[int, Path]:
    """Expose an authenticated proxy, never the actual Web UI, to the tunnel."""
    proxy_script = Path(__file__).with_name("access_proxy.py")
    proxy_log = LOCAL_ROOT / "access-proxy.log"
    env = os.environ.copy()
    env["ANIME_STUDIO_ORIGIN_PORT"] = str(origin_port)
    env["ANIME_STUDIO_PROXY_PORT"] = str(ACCESS_PROXY_PORT)
    env["ANIME_STUDIO_ACCESS_PASSWORD"] = password
    env["MPLBACKEND"] = "Agg"
    try:
        check = subprocess.run([sys.executable, "-c", "import aiohttp"], capture_output=True, text=True)
        if check.returncode:
            print("Instalando aiohttp para proteger el acceso a Web UI...", flush=True)
            run([sys.executable, "-m", "pip", "install", "-q", "aiohttp>=3.10,<4"])
        with proxy_log.open("wb") as stream:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(proxy_script)],
                env=env, cwd=str(proxy_script.parent.parent), stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
    finally:
        # The password is only passed in a child environment, not in a log.
        env.pop("ANIME_STUDIO_ACCESS_PASSWORD", None)
    STATE["proxy_process"] = proc
    from studio.runtime import save_pid
    save_pid("proxy", proc.pid)
    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError("Proxy de autenticación no inició: " + tail(proxy_log))
        try:
            response = requests.get(f"http://127.0.0.1:{ACCESS_PROXY_PORT}/__studio/health", timeout=2)
            if response.status_code == 200 and response.json().get("service") == "anime-studio-proxy":
                return ACCESS_PROXY_PORT, proxy_log
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("Proxy de autenticación no respondió: " + tail(proxy_log))


def verify_tunnel_public_url(url: str, password: str, retries: int = 10) -> bool:
    """Require the edge to reach OUR login proxy before claiming readiness."""
    if not url.startswith("https://"):
        return False
    for _ in range(retries):
        try:
            with requests.Session() as client:
                health = client.get(url + "/__studio/health", timeout=6)
                if health.status_code != 200 or health.json().get("service") != "anime-studio-proxy":
                    time.sleep(3)
                    continue
                page = client.get(url + "/__studio/login", timeout=6)
                if page.status_code != 200 or "Anime Colab Studio" not in page.text:
                    time.sleep(3)
                    continue
                signed = client.post(url + "/__studio/login", data={"password": password},
                                     allow_redirects=False, timeout=6)
                if signed.status_code != 302 or "anime_studio_session=" not in signed.headers.get("Set-Cookie", ""):
                    time.sleep(3)
                    continue
                authenticated = client.get(url + "/", timeout=12)
                if authenticated.status_code == 200 and "text/html" in authenticated.headers.get("Content-Type", ""):
                    return True
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3)
    return False


def stop_previous() -> None:
    for key in ("tunnel_process", "proxy_process", "ui_process"):
        proc = STATE.get(key)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        STATE[key] = None

def make_dashboard() -> None:
    import ipywidgets as widgets
    from IPython.display import display
    mount_drive()
    current = load_settings()
    heading = widgets.HTML("<h2>Anime Colab Studio</h2><p>Modelo, Web UI, túnel, LoRAs y archivos persistentes.</p>")
    model = widgets.Dropdown(options=list(MODELS), value=current["model"], description="Modelo:", layout=widgets.Layout(width="98%"), style={"description_width": "100px"})
    ui = widgets.ToggleButtons(options=["ComfyUI", "AUTOMATIC1111"], value=current["ui"], description="Interfaz:")
    tunnel = widgets.ToggleButtons(options=["Cloudflare", "ngrok", "Local only"], value=current["tunnel"], description="Túnel:")
    email = widgets.Text(value=current["allowed_email"], description="Tu correo:", placeholder="usuario@gmail.com", layout=widgets.Layout(width="98%"), style={"description_width": "100px"})
    save_images = widgets.Checkbox(value=current["save_to_drive"], description="Guardar imágenes PNG automáticamente en Google Drive", indent=False)
    cache = widgets.Checkbox(value=current["cache_models"], description="Guardar modelos y LoRAs en Drive (requiere ~7 GB/modelo)", indent=False)
    adetailer = widgets.Checkbox(value=current["adetailer"], description="Instalar ADetailer para AUTOMATIC1111 (caras y detalles)", indent=False)
    profile = widgets.Dropdown(options=["Balanced", "High quality", "Low VRAM"], value=current["profile"], description="Calidad:")
    choices = widgets.SelectMultiple(options=list(LORAS), value=tuple(v for v in current["loras"] if v in LORAS), description="LoRAs:", rows=5, layout=widgets.Layout(width="98%"), style={"description_width": "100px"})
    custom = widgets.Textarea(value=current["custom_loras"], description="Extra:", placeholder="Una URL https://... o hf://owner/repo/modelo.safetensors por línea", rows=3, layout=widgets.Layout(width="98%"), style={"description_width": "100px"})
    note = widgets.HTML()
    def refresh(*_):
        family = MODELS[model.value][2]
        compatible = [k for k, v in LORAS.items() if family in v[2]]
        choices.options = compatible
        note.value = f"<small>LoRAs compatibles sugeridos para <b>{family}</b>. Evita mezclar LoRAs Pony, SDXL e Illustrious sin comprobar compatibilidad.</small>"
    model.observe(refresh, names="value")
    refresh()
    valid_saved = [x for x in current["loras"] if x in list(choices.options)]
    choices.value = tuple(valid_saved)
    def get_config():
        return {"model": model.value, "ui": ui.value, "tunnel": tunnel.value, "allowed_email": email.value.strip(),
                "save_to_drive": bool(save_images.value), "cache_models": bool(cache.value), "profile": profile.value,
                "adetailer": bool(adetailer.value),
                "loras": list(choices.value), "custom_loras": custom.value.strip()}
    status = widgets.HTML("<b>Preparado.</b> Configura y pulsa Iniciar.")
    logs = widgets.Output(layout=widgets.Layout(border="1px solid #555", max_height="290px", overflow="auto"))
    save_button = widgets.Button(description="Guardar preferencias", button_style="")
    start_button = widgets.Button(description="Instalar e iniciar", button_style="success", icon="play")
    stop_button = widgets.Button(description="Detener servidor y túnel", button_style="warning", icon="stop")
    def save_click(_):
        save_settings(get_config())
        status.value = "<b>Preferencias guardadas en Google Drive.</b>"
    save_button.on_click(save_click)
    def stop_click(_):
        stop_previous()
        status.value = "<b>Procesos detenidos. Las imágenes en Drive permanecen guardadas.</b>"
    stop_button.on_click(stop_click)
    def job():
        cfg = get_config()
        start_button.disabled = True
        def log(message):
            logs.append_stdout(str(message) + "\n")
            status.value = "<b>" + str(message).replace("<", "&lt;") + "</b>"
        try:
            save_settings(cfg)
            check_gpu()
            log("Instalando interfaz...")
            app = install_ui(cfg["ui"])
            repo, filename, _ = MODELS[cfg["model"]]
            store = (DRIVE_ROOT if cfg["cache_models"] else LOCAL_ROOT)
            checkpoints = store / "checkpoints"
            loras_folder = store / "loras"
            token = secret("HF_TOKEN") or None
            log(f"Descargando/verificando checkpoint: {cfg['model']}")
            checkpoint = download_hf(repo, filename, checkpoints, token)
            if checkpoint.stat().st_size < 1_000_000_000:
                raise ValueError("Checkpoint demasiado pequeño: podría estar incompleto.")
            safe_link(checkpoint, model_dir(app, cfg["ui"]) / filename)
            installed = []
            for selection in cfg["loras"]:
                if selection not in LORAS:
                    continue
                repo_lora, file_lora, compatible, _ = LORAS[selection]
                if MODELS[cfg["model"]][2] not in compatible:
                    continue
                log(f"Descargando LoRA: {selection}")
                file = download_hf(repo_lora, file_lora, loras_folder, token)
                safe_link(file, lora_dir(app, cfg["ui"]) / file.name)
                installed.append(file.name)
            for item in cfg["custom_loras"].splitlines():
                if not item.strip() or item.strip().startswith("#"):
                    continue
                log(f"Descargando LoRA personalizado: {item[:45]}...")
                file = download_custom_lora(item, loras_folder)
                safe_link(file, lora_dir(app, cfg["ui"]) / file.name)
                installed.append(file.name)
            STATE["installed_loras"] = installed
            if cfg["ui"] == "AUTOMATIC1111" and cfg["adetailer"]:
                extension = app / "extensions" / "adetailer"
                if not (extension / "scripts").exists():
                    log("Instalando ADetailer...")
                    run(["git", "clone", "--depth", "1", "https://github.com/Bing-su/adetailer.git", str(extension)])
            destination = (DRIVE_ROOT if cfg["save_to_drive"] else LOCAL_ROOT) / "outputs" / cfg["model"].replace(" ", "_")
            destination.mkdir(parents=True, exist_ok=True)
            stop_previous()
            log("Iniciando motor de generación...")
            port, app_log = launch_ui(app, cfg, destination)
            wait_for_http(port, STATE["ui_process"], app_log)
            log("Creando acceso web privado...")
            url = launch_tunnel(cfg["tunnel"], port, cfg["allowed_email"])
            status.value = f'<b>Servidor listo:</b> <a href="{url}" target="_blank">{url}</a><br/>Imágenes: <code>{destination}</code>'
            logs.append_stdout("Interfaz: " + cfg["ui"] + "\n")
            logs.append_stdout("Modelo: " + filename + "\n")
            logs.append_stdout("LoRAs instalados: " + (", ".join(installed) or "(ninguno)") + "\n")
            if cfg["ui"] == "AUTOMATIC1111" and installed:
                logs.append_stdout("Usa en el prompt: " + " ".join(f"<lora:{Path(x).stem}:0.7>" for x in installed) + "\n")
            if cfg["ui"] == "ComfyUI":
                logs.append_stdout("Workflow listo en ComfyUI > Workflows > AnimeStudio y en Drive/AnimeColabStudio/AnimeStudio-workflow.json\n")
        except Exception as error:
            logs.append_stdout(f"ERROR ({type(error).__name__}): {error}\n")
            status.value = "<b>Error de inicio. Consulta el registro y vuelve a intentar.</b>"
        finally:
            start_button.disabled = False
    def start_click(_):
        logs.clear_output()
        status.value = "<b>Iniciando...</b>"
        threading.Thread(target=job, daemon=True).start()
    start_button.on_click(start_click)
    display(widgets.VBox([heading, model, ui, tunnel, email, profile, save_images, cache, adetailer,
                          widgets.HTML("<b>LoRAs preseleccionados (opcional)</b>"), choices,
                          widgets.HTML("<b>LoRAs personalizados (opcional)</b>"), custom, note,
                          widgets.HBox([save_button, start_button, stop_button]), status, logs]))
    print("Secretos opcionales en Colab > icono de llave: HF_TOKEN, NGROK_AUTHTOKEN, CIVITAI_API_TOKEN, WEBUI_PASSWORD.")
    print("Nunca guardamos contraseñas ni tokens en Google Drive.")

if __name__ == "__main__":
    make_dashboard()
