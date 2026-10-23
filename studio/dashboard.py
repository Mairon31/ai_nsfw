"""Responsive dashboard for Google Colab: persisted jobs + visible logs."""
from __future__ import annotations

import html
import threading
import time
import traceback

import ipywidgets as widgets
from IPython.display import display

from studio import launcher, runtime


def make_dashboard():
    # Mounting Drive uses the *notebook* context; do not do it in a detached worker.
    launcher.mount_drive()
    values = launcher.load_settings()

    title = widgets.HTML(
        "<h2>Anime Colab Studio</h2>"
        "<p>Selecciona el modelo, interfaz, túnel y LoRAs. "
        "La instalación se ejecuta aparte y puedes detenerla o reiniciarla.</p>")
    layout = widgets.Layout(width="98%")
    desc = {"description_width": "112px"}
    model = widgets.Dropdown(options=list(launcher.MODELS), value=values["model"],
                             description="Modelo", layout=layout, style=desc)
    interface = widgets.ToggleButtons(options=["ComfyUI", "AUTOMATIC1111"],
                                      value=values["ui"], description="Interfaz")
    tunnel = widgets.ToggleButtons(options=["Cloudflare", "ngrok", "Local only"],
                                   value=values["tunnel"], description="Túnel")
    email = widgets.Text(value=values["allowed_email"], description="Tu correo",
                         placeholder="correo@gmail.com", layout=layout, style=desc)
    quality = widgets.Dropdown(options=["Balanced", "High quality", "Low VRAM"],
                               value=values["profile"], description="Calidad", style=desc)
    save_images = widgets.Checkbox(
        value=values["save_to_drive"],
        description="Guardar imágenes PNG automáticamente en Google Drive",
        indent=False)
    cache = widgets.Checkbox(
        value=values["cache_models"],
        description="Conservar modelos y LoRAs en Google Drive para próximos inicios",
        indent=False)
    detail = widgets.Checkbox(
        value=values.get("adetailer", False),
        description="Instalar ADetailer (AUTOMATIC1111)",
        indent=False)
    choices = widgets.SelectMultiple(
        options=list(launcher.LORAS),
        description="LoRAs", rows=5, layout=layout, style=desc)
    custom = widgets.Textarea(
        value=values.get("custom_loras", ""), description="Extra",
        placeholder="hf://usuario/repo/archivo.safetensors o URL por línea",
        rows=3, layout=layout, style=desc)
    compatible_note = widgets.HTML()
    def model_changed(*args):
        family = launcher.MODELS[model.value][2]
        active = set(choices.value)
        available = [name for name, value in launcher.LORAS.items() if family in value[2]]
        choices.options = available
        choices.value = tuple(name for name in available if name in active)
        compatible_note.value = (
            f"<small>Familia: <b>{html.escape(family)}</b>. "
            "Los LoRAs sugeridos se filtran para evitar mezclas incompatibles.</small>")
    model.observe(model_changed, names="value")
    model_changed()
    choices.value = tuple(name for name in values["loras"] if name in choices.options)

    progress = widgets.IntProgress(
        value=0, min=0, max=100, description="Progreso",
        layout=widgets.Layout(width="98%"))
    status = widgets.HTML("<b>Preparado para iniciar.</b>")
    logs = widgets.HTML(runtime.render_log())
    last = widgets.HTML(f"<small>Registro: <code>{runtime.LOG}</code> — se actualiza automáticamente.</small>")
    save_btn = widgets.Button(description="Guardar preferencias", icon="save")
    start_btn = widgets.Button(description="Instalar / iniciar", icon="play", button_style="success")
    stop_btn = widgets.Button(description="Detener / cancelar", icon="stop", button_style="warning")
    refresh_btn = widgets.Button(description="Actualizar estado y logs", icon="refresh")
    help_text = widgets.HTML(
        "<small>En Colab, los secretos opcionales se configuran en el icono de llave: "
        "<code>HF_TOKEN</code>, <code>NGROK_AUTHTOKEN</code>, "
        "<code>CIVITAI_API_TOKEN</code> y <code>WEBUI_PASSWORD</code>. "
        "Aunque cierres el navegador, el progreso se conserva en el registro mientras el runtime siga activo.</small>")

    def get_config():
        return {
            "model": model.value, "ui": interface.value, "tunnel": tunnel.value,
            "allowed_email": email.value.strip(), "profile": quality.value,
            "save_to_drive": bool(save_images.value),
            "cache_models": bool(cache.value),
            "adetailer": bool(detail.value),
            "loras": list(choices.value),
            "custom_loras": custom.value.strip(),
        }

    def update_display():
        data = runtime.snapshot()
        stage = data.get("stage", "idle")
        progress.value = int(data.get("percent", 0))
        progress.bar_style = (
            "success" if stage == "ready" else "danger" if stage == "error"
            else "warning" if stage == "stopped" else "info")
        message = html.escape(str(data.get("message", "Sin actividad.")))
        updated = html.escape(data.get("updated_at", ""))
        active = runtime.process_active()
        url = data.get("url")
        link = ""
        if stage == "ready" and url:
            safe_url = html.escape(str(url), quote=True)
            if str(url).startswith("https://"):
                link = f'<p><a href="{safe_url}" target="_blank" rel="noopener"><b>Abrir Web UI</b></a></p>'
            else:
                link = f"<p><code>{safe_url}</code></p>"
        status.value = (f"<b>{message}</b>{link}"
                        f"<small>Etapa: {html.escape(stage)}"
                        f" · Último cambio: {updated or 'sin datos'}</small>")
        start_btn.disabled = active
        stop_btn.disabled = not active
        logs.value = runtime.render_log(50)

    def save_click(_):
        try:
            launcher.save_settings(get_config())
            status.value = "<b>Preferencias guardadas en Google Drive.</b>"
        except Exception as exc:
            status.value = f"<b>Error guardando ajustes:</b> {html.escape(str(exc))}"
    save_btn.on_click(save_click)

    def start_click(_):
        try:
            config = get_config()
            launcher.save_settings(config)
            environment_secrets = {
                key: launcher.secret(key) for key in
                ("HF_TOKEN", "NGROK_AUTHTOKEN", "CIVITAI_API_TOKEN", "WEBUI_PASSWORD")
            }
            runtime.launch_worker(config, environment_secrets)
        except Exception as exc:
            status.value = f"<b>No se pudo iniciar:</b> {html.escape(str(exc))}"
        finally:
            update_display()
    start_btn.on_click(start_click)

    def stop_click(_):
        try:
            runtime.stop_session()
        except Exception as exc:
            status.value = f"<b>No se pudo detener:</b> {html.escape(str(exc))}"
        finally:
            update_display()
    stop_btn.on_click(stop_click)

    refresh_btn.on_click(lambda _: update_display())

    display(widgets.VBox([
        title, model, interface, tunnel, email, quality, save_images, cache, detail,
        widgets.HTML("<b>LoRAs opcionales</b>"), choices,
        widgets.HTML("<b>LoRAs personalizados (uno por línea)</b>"),
        custom, compatible_note,
        widgets.HBox([save_btn, start_btn, stop_btn, refresh_btn]),
        status, progress, last, logs, help_text,
    ]))

    def poll():
        # ipywidgets updates may be throttled on some Colab frontends.
        # "Actualizar estado y logs" is always available as manual fallback.
        while True:
            try:
                update_display()
            except Exception:
                pass
            time.sleep(2)
    threading.Thread(target=poll, daemon=True, name="anime-studio-monitor").start()
    update_display()
