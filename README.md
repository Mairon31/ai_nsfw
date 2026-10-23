# Anime Colab Studio

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mairon31/ai_nsfw/blob/master/Anime_Studio_Colab.ipynb)

**Un solo notebook interactivo para lanzar modelos anime de Stable Diffusion XL en Google Colab**, con interfaz, túnel protegido y persistencia de preferencias e imágenes en Google Drive.

## Comenzar

1. Haz clic en **Open In Colab**.
2. Selecciona **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.
3. Ejecuta las **dos celdas** (o **Ejecutar todo**) y autoriza Google Drive.
4. Escoge **modelo, ComfyUI/AUTOMATIC1111, Cloudflare/ngrok, calidad y LoRAs**.
5. Escribe el **correo Google autorizado para abrir la interfaz remota**.
6. Pulsa **Instalar e iniciar**. El notebook instala herramientas, descarga el checkpoint y abre una URL privada. Elige solo modelos y contenidos permitidos por el proveedor.

En ejecuciones posteriores los ajustes se cargan desde Google Drive. El botón **Guardar preferencias** permite actualizarlos sin iniciar una instalación.

## Modelos disponibles

| Modelo | Archivo/Origen | Familia |
| --- | --- | --- |
| NoobAI-XL 1.0 | [Laxhar/noobai-XL-1.0](https://huggingface.co/Laxhar/noobai-XL-1.0) | NoobAI XL |
| Pony Diffusion V6 XL | [6chan/Pony-Diffusion-V6-XL](https://huggingface.co/6chan/Pony-Diffusion-V6-XL) | Pony |
| WAI-Illustrious SDXL v17 | [yufusoft/WAI-illustrious-SDXL](https://huggingface.co/yufusoft/WAI-illustrious-SDXL) | Illustrious |
| WAI-NSFW-Illustrious v12 (legacy) | [TheReprinter/WAI-NSFW-illustrious-SDXL](https://huggingface.co/TheReprinter/WAI-NSFW-illustrious-SDXL) | Illustrious |
| Animagine XL 4.0 Opt | [cagliostrolab/animagine-xl-4.0](https://huggingface.co/cagliostrolab/animagine-xl-4.0) | SDXL |
| Illustrious XL | [drawhisper/illustrious-xl](https://huggingface.co/drawhisper/illustrious-xl) | Illustrious |

El notebook descarga el checkpoint de la opción elegida, **no todos los modelos**. Repositorios de terceros pueden cambiar o exigir aceptar licencias.

## LoRAs

El selector ofrece opciones filtradas por familia:

- **Anime Style LoRA SDXL:** estilo anime general para checkpoints SDXL, con compatibilidad variable según derivado.
- **Sensual Face v3 (tipo 1/2):** para la familia Illustrious, incluyendo WAI.
- **Sensual Face v2 (tipo 1/2):** para la familia Pony.

Puedes introducir **uno por línea** LoRAs personalizados desde Hugging Face y Civitai:

~~~text
hf://USUARIO/REPOSITORIO/modelo.safetensors
https://huggingface.co/USUARIO/REPOSITORIO/resolve/main/modelo.safetensors
https://civitai.com/api/download/models/123456
~~~

El gestor instala todos los LoRAs seleccionados. En **AUTOMATIC1111** se muestran etiquetas para insertar en el prompt; en **ComfyUI** se genera un workflow con todos los LoRAs encadenados. La compatibilidad de un LoRA externo la debe comprobar el usuario antes de combinarlo.

## Google Drive y recuperación

Se usa la carpeta **Mi unidad/AnimeColabStudio/**:

~~~text
AnimeColabStudio/
├── settings.json                # Modelo, interfaz, túnel, preferencias (sin tokens)
├── checkpoints/                 # Pesos si activas cache_models
├── loras/                       # LoRAs si activas cache_models
├── outputs/
│   └── NOMBRE_DEL_MODELO/      # PNG originales sin pérdida
└── AnimeStudio-workflow.json    # Workflow exportado de ComfyUI
~~~

El guardado de imágenes en Drive viene **activado por defecto**. ComfyUI apunta directamente a este directorio mediante \`--output-directory\` y AUTOMATIC1111 tiene sus carpetas de salida configuradas en \`config.json\`. Las imágenes PNG mantienen resolución nativa y metadatos cuando la UI los soporta.

**Persistencia:** tras un reinicio del runtime, las preferencias e imágenes de Drive permanecen disponibles. Los pesos se conservarán también **solo si activas la caché de modelos en Drive**. La GPU, la instalación de UI y el enlace del túnel son temporales y deben volver a iniciarse.

## Seguridad y secretos

En Colab, abre **Secretos** (icono de llave) y permite el acceso del notebook a los secretos necesarios:

| Secreto | Cuándo se usa |
| --- | --- |
| \`NGROK_AUTHTOKEN\` | Túnel ngrok |
| \`HF_TOKEN\` | Checkpoints/LoRAs privados, limitados o gated de Hugging Face |
| \`CIVITAI_API_TOKEN\` | Descargas de LoRAs con autenticación |
| \`WEBUI_PASSWORD\` | Contraseña adicional para AUTOMATIC1111 (usuario: \`studio\`) |

- **Cloudflare** usa \`--allowed-mail\` para restringir el acceso al correo configurado.
- **ngrok** emplea una política OAuth de Google que permite únicamente el correo configurado.
- **No se almacenan tokens ni contraseñas en el repositorio o en \`settings.json\`.**
- Evita publicar las URLs de tus túneles. Usa exclusivamente tus propios contenidos o contenidos que tengas derecho a procesar.

## Ajustes de calidad

| Preset | Resolución inicial | Pasos | CFG |
| --- | --- | --- | --- |
| Balanced | 832 × 1216 | 28 | 6 |
| High quality | 1024 × 1536 | 36 | 6 |
| Low VRAM | 768 × 1024 | 22 | 5 |

Para subir aún más la calidad, utiliza el upscaler o una segunda pasada dentro de la Web UI. El notebook prioriza el guardado **PNG sin compresión con pérdida** en lugar de prometer una calidad universal; el resultado depende del modelo, prompts, LoRAs y GPU.

## Problemas comunes

- **No hay GPU:** en Colab selecciona GPU desde las opciones del runtime.
- **El checkpoint no descarga:** comprueba que el repositorio sigue accesible y configura \`HF_TOKEN\` si es gated.
- **El túnel no aparece:** comprueba el correo autorizado, la clave de ngrok (cuando aplique) y los registros impresos.
- **AUTOMATIC1111 no inicia:** consulta el registro en \`/content/AnimeColabStudio/automatic1111-server.log\`. Algunas versiones de Colab/Python/Torch pueden requerir ajustes manuales.
- **No sale un LoRA en la interfaz:** comprueba familia/modelo, refresca la lista o reinicia el servicio.
- **Google Drive lento:** desactiva la caché de modelos si no necesitas persistencia de pesos; el guardado de resultados puede seguir activado.

## Validación y limitaciones

Los tests de CI comprueban el catálogo, persistencia sin secretos, filtros de URL, estructura del workflow y formato del notebook. No equivalen a una **prueba real de inferencia GPU**, que requiere una sesión de Colab con recursos disponibles.

El acceso Web UI en un entorno administrado de Colab está sujeto a sus términos y a los límites aplicables. Los planes pagados con unidades de cómputo positivas eliminan algunas restricciones del nivel gratuito, pero **no garantizan recursos ni permiten eludir políticas de contenido**.

## Estructura del proyecto

~~~text
Anime_Studio_Colab.ipynb      # Abre esto en Google Colab
studio/launcher.py            # Descargas, panel, persistencia y procesos
tests/test_launcher.py        # Pruebas offline
.github/workflows/validate.yml
~~~
