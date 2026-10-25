# Anime Colab Studio

Un entorno de **generación de imágenes y videos con IA** desde Google Colab. Selecciona el modelo, motor e interfaz; accede a la Web UI mediante Cloudflare o ngrok con una **contraseña de sesión** (sin códigos por correo); guarda modelos y resultados en Google Drive.

**Dos formas de utilizarlo:**

| Notebook | Recomendado para | Abrir |
| --- | --- | --- |
| **Studio completo** | PC, tablet y navegadores con widgets de Colab; panel interactivo, logs y controles | [Abrir en Colab](https://colab.research.google.com/github/Mairon31/ai_nsfw/blob/master/Anime_Studio_Colab.ipynb) |
| **Studio móvil** | iPhone y Android; formularios nativos de Colab, sin widgets, selección de imagen/video y diagnóstico táctil | [Abrir versión móvil](https://colab.research.google.com/github/Mairon31/ai_nsfw/blob/master/Anime_Studio_Mobile.ipynb) |

[![Validate](https://github.com/Mairon31/ai_nsfw/actions/workflows/validate.yml/badge.svg)](https://github.com/Mairon31/ai_nsfw/actions/workflows/validate.yml)

> **Alcance de las pruebas:** CI valida Python, configuraciones, flujos locales simulados y el funcionamiento de rutas seleccionadas. Una ejecución real de inferencia en GPU, cada checkpoint de Hugging Face y los enlaces temporales Cloudflare/ngrok **no están garantizados** por CI. Revisa los logs de tu sesión.

## Características

- **Imágenes:** NoobAI-XL, Pony Diffusion V6 XL, WAI-Illustrious, WAI-NSFW-Illustrious, Animagine XL e Illustrious XL.
- **Video, opción 1:** Wan 2.2 TI2V 5B en **ComfyUI**. Descarga *solo cuando lo seleccionas* el transformer, UMT5, VAE y workflow oficiales para texto/imagen a video.
- **Video, opción 2:** **WanGP** (proyecto de DeepBeepMeep). Una sola Web UI con familias Wan 2.1, Wan 2.2, LTX-2 y HunyuanVideo, descarga bajo demanda, colas, presets, LoRAs y herramientas de procesamiento. **El modelo final se elige dentro de WanGP; el selector del notebook funciona como recomendación de familia.**
- **Interfaz:** ComfyUI y AUTOMATIC1111 para imágenes; ComfyUI y WanGP para video.
- **LoRAs:** preseleccionados y personalizados para familias de imágenes compatibles; WanGP administra sus propios LoRAs de video.
- **Persistencia:** checkpoints y LoRAs de imágenes en Drive si activas caché; videos de ComfyUI, checkpoints de WanGP, LoRAs y sus carpetas de salida/workspaces en Drive si habilitas las opciones.
- **Supervisión:** avance por etapa, registros, botón de cancelación y opción de celda ejecutándose mientras un servicio real está activo; no impide que Colab finalice tu VM.

## Inicio rápido desde PC o tablet

1. Abre el [notebook completo](https://colab.research.google.com/github/Mairon31/ai_nsfw/blob/master/Anime_Studio_Colab.ipynb).
2. En **Entorno de ejecución → Cambiar tipo**, activa una **GPU NVIDIA**. La disponibilidad de T4, L4 o A100 depende de Colab.
3. Ejecuta la **celda 1** (dependencias) y la **celda 2** (descarga la versión más reciente de este repositorio y monta Drive).
4. Selecciona **Imágenes** o **Videos** y sus opciones.
5. Elige **Cloudflare** para acceso remoto sin cuenta de ngrok; establece opcionalmente el secreto `WEBUI_PASSWORD` desde el icono de llave de Colab.
6. Pulsa **Instalar / iniciar** y consulta el avance. Cuando termine, abre la URL HTTPS y usa la contraseña mostrada en el notebook. No hay autenticación repetida por correo.
7. La **celda 3** ofrece diagnóstico incluso si los widgets no actualizan. La **celda 5** mantiene una celda supervisando la ejecución; la **celda 4** permite un arranque directo alternativo con logs.

El launcher reconoce instalaciones existentes en la sesión y reutiliza los archivos de Drive configurados. **No borres Google Drive** al recuperar una sesión.

## Inicio rápido desde móvil (iPhone / Android)

1. Abre el [notebook para móvil](https://colab.research.google.com/github/Mairon31/ai_nsfw/blob/master/Anime_Studio_Mobile.ipynb) en **Chrome o Safari**, con tu cuenta Google.
2. Activa GPU en el runtime. Para menús que no aparezcan, utiliza la opción **Solicitar sitio de escritorio** del navegador.
3. Ejecuta **celda 1** para preparar bibliotecas y montar Drive. Autoriza el acceso solicitado por Google.
4. En **celda 2**, selecciona el modo, modelo, interfaz, túnel, calidad, LoRA opcional y almacenamiento. Ejecuta la celda. Esta vista utiliza `# @param` (formularios de Colab), no `ipywidgets`.
5. Ejecuta la **celda 3** y actualízala cuando quieras: muestra progreso, URL, contraseña y últimos logs. Abre la interfaz en otra pestaña.
6. **Celda 4:** monitor continuo opcional. Mantiene código real en ejecución y comprueba la API. Puedes detener la celda sin detener el motor; para apagar el servidor activa `DETENER_SERVIDOR` en la celda 3.

El teléfono no necesita GPU; toda la inferencia sucede en la VM de Colab. No dependas de dejar Safari/Chrome en segundo plano para mantener viva una sesión: iOS/Android pueden suspender las pestañas y Colab tiene límites propios.

## Modelos y motores de video

| Selección | Motor | Cómo se obtiene | Qué probar |
| --- | --- | --- | --- |
| **Wan 2.2 TI2V 5B** | ComfyUI | Pesos oficiales (~10 GB transformer, encoder ~6–7 GB y VAE ~1–2 GB) y workflow de ComfyUI | Texto a video o imagen a video; empieza en 480p y clips cortos |
| **Wan 2.2 T2V / I2V** | WanGP | Selecciona el modelo específico en la Web UI; descarga cuando haga falta | Movimiento complejo, modelos 14B, uso de offloading |
| **Wan 2.1 1.3B** | WanGP | Desde el catálogo interno | Pruebas rápidas con menos memoria |
| **LTX-2.x** | WanGP | Desde el catálogo interno | Animación y variantes de video; consulta requisitos concretos |
| **HunyuanVideo** | WanGP | Desde el catálogo interno | Video con cuantización y perfil de memoria bajo |

**Recomendaciones importantes:**

- En **T4 (normalmente 16 GB)** empieza con Wan 2.1 1.3B o Wan 2.2 TI2V 5B en ComfyUI con offload. La CPU/RAM, GPU, frames y resolución afectan mucho al tiempo.
- Para **modelos grandes (14B)** considera **L4 o A100**, cuantización y offload, si la VM tiene suficiente RAM. No asumas que un T4 ejecutará todos los presets WanGP.
- Usa pruebas de **3–5 segundos** (por ejemplo 49–81 frames) antes de pedir clips largos. Una animación final de 30–60 s suele requerir varias tomas y edición o continuación dentro del backend.
- El selector WanGP del notebook **no instala automáticamente un modelo de decenas de GB**: la aplicación muestra sus variantes y permite escoger según tu GPU.
- Wan 2.2 TI2V 5B en ComfyUI descarga sus tres componentes la primera vez. En ComfyUI abre **Workflows → AnimeStudio-Wan22-TI2V-5B**. Si guardas en Drive también habrá una copia en `AnimeColabStudio/workflows/`.

**Fuentes de las implementaciones:** [WanGP oficial](https://github.com/DeepBeepMeep/Wan2GP), [Guía CLI WanGP](https://github.com/DeepBeepMeep/Wan2GP/blob/main/docs/CLI.md), [Wan 2.2 Comfy-Org](https://huggingface.co/Comfy-Org/Wan-2.2) y [workflow oficial TI2V 5B](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_wan2_2_5B_ti2v.json).

## Directorios persistentes en Google Drive

Cuando usas las opciones de guardar y caché:

```text
Mi unidad/
└── AnimeColabStudio/
    ├── settings.json                 # preferencias no sensibles
    ├── checkpoints/                 # modelos de imágenes descargados
    ├── loras/                       # LoRAs de imágenes
    ├── video-checkpoints/
    │   ├── diffusion_models/        # Wan 2.2 5B
    │   ├── text_encoders/
    │   └── vae/
    ├── video-cache/
    │   └── wangp/
    │       ├── ckpts/              # modelos bajo demanda de WanGP
    │       ├── loras/
    │       └── loras_i2v/
    ├── videos/                      # renders, workspaces y ajustes de WanGP
    ├── outputs/                     # PNG de generación de imágenes
    └── workflows/                   # workflows ComfyUI de video
```

Los paquetes Python, clones y túneles están en la VM temporal de Colab y pueden requerir instalación tras un reinicio. Los archivos guardados realmente en Drive sobreviven. **El caché de video puede ocupar decenas de GB**; revisa tu cuota de Drive antes de instalar modelos grandes.

## Seguridad y acceso remoto

El launcher mantiene el motor escuchando en `127.0.0.1`, publica **solo un proxy de acceso**, y usa cookies seguras para HTTP y WebSocket. No utiliza `--allowed-mail` ni solicita códigos repetidos por correo. Puedes poner `WEBUI_PASSWORD` en Secretos de Colab; sin ello, genera una contraseña nueva por sesión y la muestra en el notebook.

Secretos opcionales (icono de llave): `HF_TOKEN` para repositorios privados o con permisos, `NGROK_AUTHTOKEN` para ngrok, `WEBUI_PASSWORD` para controlar el acceso. No se guardan esos valores en GitHub ni en `settings.json`. No compartas el notebook con las salidas visibles cuando tengan URL y contraseña, ni publiques la URL del túnel.

## Diagnóstico y recuperación

- **Estado y logs:** celda 3; archivos en `/content/AnimeColabStudio/runtime.log` y `*-server.log`. La versión móvil permite verlos sin widgets.
- **Celda completada:** es normal para la celda 2; el servidor corre como proceso supervisado. Puedes usar la celda de monitor para mostrar ejecución activa.
- **Error al arrancar A1111:** se usa Python 3.10 aislado, workaround `setuptools`/CLIP y mirror de Stable Diffusion. No modifiques globalmente Python 3.13 del notebook.
- **Conflictos CUDA/VRAM:** empieza con un modelo más ligero, menos frames, 480p, batch 1, perfil de memoria 4 (WanGP); comprueba la GPU disponible en Colab.
- **Espacio insuficiente:** elimina solo cachés prescindibles de la VM o checkpoints que no utilices. No borres carpetas de Drive sin revisar su contenido.
- **Interfaz no abre:** comprueba `ready` y verificación externa en diagnóstico; si el túnel caduca, detén, actualiza e inicia otra sesión. Los enlaces son temporales.
- **Sin respuesta después de una actualización:** vuelve a ejecutar la celda 2 (o 1 en móvil) para traer la última versión del repositorio.
- **Colab se apaga:** el servicio impone una vida máxima de VM, límites de saldo y posible desconexión por inactividad. **Ningún monitor o técnica de mantener la celda abierta los elimina**. Consulta las [FAQ de Colab](https://research.google.com/colaboratory/faq.html).

## Desarrollo y pruebas

```bash
python -m pip install ipywidgets huggingface_hub requests 'aiohttp>=3.10,<4'
python -m compileall -q studio tests
python -m unittest discover -s tests -v
```

GitHub Actions valida el código y algunas dependencias externas, pero no genera MP4 ni PNG en una GPU real. Los cambios están pensados para no romper los modos existentes de imágenes. No distribuyas contenidos que vulneren derechos de terceros ni el reglamento del proveedor de cómputo.

---

**English quick summary:** Open either Colab notebook above. Select image generation (ComfyUI/AUTOMATIC1111) or video (Wan 2.2 5B ComfyUI / WanGP). The mobile notebook uses native Colab forms. Tunnel access is password-protected without email OTP. Cache and outputs can be stored on Google Drive. GPU support and session duration depend on Colab; CUDA video inference has not been end-to-end tested in CI.
