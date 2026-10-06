# nude-image-detector

API REST en **Python + Flask** que evalúa si una imagen contiene desnudos. Acepta una **imagen
subida** o la **URL de una imagen** y devuelve un veredicto (`block` / `review` / `allow`) con el
riesgo detectado, para que puedas **rechazar la subida antes de guardarla** en tus servidores.

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/analyze" -F "file=@foto.jpg"
```

```jsonc
{
  "verdict": "block",            // block | review | allow
  "nsfw": true,                  // true cuando verdict == "block"
  "risk_score": 0.9288,          // 0..1
  "reasons": ["FEMALE_BREAST_EXPOSED: pecho femenino expuesto (confianza 0.94, área 6.1%, peso 0.85)"],
  "detections": [{ "label": "FEMALE_BREAST_EXPOSED", "confidence": 0.9412, "box": [412,180,733,540] }],
  "image": { "width": 1200, "height": 800, "sha256": "a8ce5da1…" },
  "elapsed_ms": 41.7
}
```

---

## Instalación y arranque

Requisitos: **Python 3.10+**. No hace falta GPU ni descargar pesos: el modelo ONNX (12 MB) viene
dentro del paquete `nudenet` instalado desde PyPI.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# desarrollo
.venv/bin/python run.py

# producción
.venv/bin/gunicorn -c gunicorn.conf.py wsgi:app
```

Comprueba que el modelo cargó:

```bash
curl -s http://127.0.0.1:8000/ready
# {"ready":true,"engine":"nudenet","model":"320n@320","version":"3.4.2"}
```

---

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/v1/analyze` | Analiza **una** imagen (archivo, URL, base64 o binario) |
| `POST` | `/v1/analyze/batch` | Analiza **varias** imágenes; un fallo individual no aborta el lote |
| `GET` | `/v1/info` | Modelo, perfil de moderación y límites activos |
| `GET` | `/v1/labels` | Catálogo de etiquetas con severidad y peso |
| `GET` | `/health` | El proceso responde |
| `GET` | `/ready` | 200 si el modelo está listo, 503 si no |
| `DELETE` | `/v1/cache` | Vacía la caché (requiere `NID_API_KEYS`) |

### `POST /v1/analyze`

Cuatro formas de enviar la imagen:

```bash
HOST=http://127.0.0.1:8000

# 1) Archivo (multipart, campo "file")
curl -s -X POST "$HOST/v1/analyze" -F "file=@foto.jpg"

# 2) URL de una imagen
curl -s -X POST "$HOST/v1/analyze" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://ejemplo.com/foto.jpg"}'

# 3) Imagen en base64 (admite data URLs)
curl -s -X POST "$HOST/v1/analyze" \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$(base64 -w0 foto.jpg)\"}"

# 4) Cuerpo binario crudo
curl -s -X POST "$HOST/v1/analyze" -H 'Content-Type: image/jpeg' --data-binary @foto.jpg
```

Parámetros opcionales (por *query string*, formulario o JSON):

| Parámetro | Valores | Efecto |
|---|---|---|
| `strictness` | `strict` \| `balanced` \| `lenient` | Perfil de moderación |
| `block_threshold` | 0–1 | Sobrescribe el umbral de bloqueo |
| `review_threshold` | 0–1 | Sobrescribe el umbral de revisión |
| `min_confidence` | 0–1 | Confianza mínima para que una detección puntúe |
| `censor` | bool | Añade una copia pixelada/difuminada en base64 si no pasa el filtro |
| `no_detections` | bool | Omite la lista de detecciones (respuesta más ligera) |
| `reject_on_block` | bool | Devuelve **HTTP 422** en vez de 200 cuando bloquea |

**Un análisis correcto siempre devuelve `200 OK`**, incluso si el veredicto es `block`: la API hizo
su trabajo. Usa `?reject_on_block=true` si tu pasarela prefiere un error HTTP para las rechazadas.

### Respuesta completa

```jsonc
{
  "request_id": "9f1c0a5b6d7e8f90",
  "verdict": "block",
  "verdict_description_es": "Rechaza la subida: se detectó desnudez con confianza suficiente.",
  "nsfw": true,
  "risk_score": 0.9288,
  "scores": { "explicit": 0.9288, "suggestive": 0.0 },
  "reasons": ["FEMALE_BREAST_EXPOSED: pecho femenino expuesto (confianza 0.94, área 6.1%, peso 0.85)"],
  "flags": ["explicit_nudity"],
  "explicit_labels": ["FEMALE_BREAST_EXPOSED"],
  "labels_found": ["FEMALE_BREAST_EXPOSED", "BELLY_EXPOSED", "FACE_FEMALE"],
  "severity_counts": { "explicit": 1, "suggestive": 1, "neutral": 1 },
  "top_label": "FEMALE_BREAST_EXPOSED",
  "max_confidence": 0.9412,
  "frames_with_findings": 1,
  "detections": [
    {
      "label": "FEMALE_BREAST_EXPOSED",
      "confidence": 0.9412,
      "severity": "explicit",          // explicit | suggestive | neutral
      "weight": 0.85,
      "box": [412, 180, 733, 540],     // x1, y1, x2, y2 en píxeles
      "box_xywh": [412, 180, 321, 360],
      "area_ratio": 0.0611,            // fracción del fotograma que ocupa
      "frame_index": 0,
      "counted": true,                 // false si quedó bajo min_confidence
      "description_es": "Pecho femenino expuesto (pezón/areola)"
    }
  ],
  "policy": { "profile": "balanced", "block_threshold": 0.55, "review_threshold": 0.28 },
  "image": {
    "format": "jpeg", "mime_type": "image/jpeg", "width": 1200, "height": 800,
    "source_width": 4000, "source_height": 2667, "bytes": 481203, "sha256": "a8ce5da1…",
    "animated": false, "frames_total": 1, "frames_analysed": 1,
    "downscaled": true, "orientation_applied": true, "mime_mismatch": false
  },
  "source": { "type": "upload", "filename": "foto.jpg", "bytes": 481203 },
  "model": { "engine": "nudenet", "model": "320n@320", "version": "3.4.2", "ready": true },
  "cached": false,
  "elapsed_ms": 41.7
}
```

### `POST /v1/analyze/batch`

```bash
curl -s -X POST "$HOST/v1/analyze/batch" \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://a.com/1.jpg","https://b.com/2.jpg"]}'
```

```jsonc
{
  "request_id": "…",
  "count": 2,
  "summary": {
    "total": 2, "succeeded": 1, "failed": 1,
    "verdicts": { "allow": 1, "review": 0, "block": 0, "error": 1 },
    "highest_risk_score": 0.04, "any_blocked": false
  },
  "results": [
    { "index": 0, "ok": true,  "result": { "verdict": "allow", "…": "…" } },
    { "index": 1, "ok": false, "error": { "code": "upstream_error", "status": 502 } }
  ],
  "elapsed_ms": 180.4
}
```

También acepta `multipart/form-data` con varias partes (`files`) o una lista mixta
`{"items":[{"url":…},{"image_base64":…}]}`. Máximo `NID_BATCH_MAX_ITEMS` elementos.

### Errores

Todas las respuestas de error comparten forma:

```json
{ "request_id": "9f1c0a5b", "error": { "code": "blocked_destination", "message": "…", "status": 403, "details": {} } }
```

Programa contra `error.code`, nunca contra el texto del mensaje.

| `code` | HTTP | Cuándo ocurre |
|---|---|---|
| `missing_image` | 400 | No llegó ninguna imagen |
| `bad_request` | 400 | Parámetros o JSON inválidos |
| `invalid_url` | 400 | URL mal formada, esquema no http(s), credenciales embebidas |
| `too_many_items` | 400 | El lote supera `NID_BATCH_MAX_ITEMS` |
| `unauthorized` | 401 | Falta o es inválida `X-API-Key` |
| `forbidden` | 403 | Operación no permitida (p. ej. URL fetching desactivado) |
| `blocked_destination` | 403 | La URL apunta a una red privada o protegida |
| `payload_too_large` | 413 | Se superó `NID_MAX_CONTENT_LENGTH` o `NID_FETCH_MAX_BYTES` |
| `unsupported_media_type` | 415 | Formato no permitido (p. ej. HEIC/AVIF) |
| `unprocessable_image` | 422 | El archivo no se pudo decodificar como imagen |
| `image_too_large` | 422 | Demasiados píxeles (posible bomba de descompresión) |
| `rate_limited` | 429 | Límite de peticiones superado (incluye `Retry-After`) |
| `upstream_error` | 502 | El servidor remoto falló o devolvió un estado inesperado |
| `upstream_timeout` | 504 | El servidor remoto tardó demasiado |
| `inference_error` | 500 | El modelo falló al procesar la imagen |
| `model_unavailable` | 503 | El motor de detección no está cargado |

---

## Consumo desde tu backend

Python:

```python
import requests

with open("foto.jpg", "rb") as fh:
    r = requests.post("http://127.0.0.1:8000/v1/analyze", files={"file": fh}, timeout=30)

r.raise_for_status()
resultado = r.json()

if resultado["verdict"] == "block":
    raise ValueError("Imagen rechazada por contenido explícito")
elif resultado["verdict"] == "review":
    enviar_a_cola_de_moderacion(resultado["image"]["sha256"])
else:
    guardar_imagen()
```

JavaScript:

```js
const fd = new FormData();
fd.append("file", fileInput.files[0]);

const r = await fetch("/v1/analyze?strictness=balanced", { method: "POST", body: fd });
const { verdict, risk_score: risk, reasons } = await r.json();

if (verdict === "block") throw new Error(`Imagen rechazada: ${reasons[0] ?? risk}`);
```

Flujo recomendado:

```
usuario sube imagen
        │
        ▼
POST /v1/analyze   ← ANTES de escribir en S3 / tu BD
        │
        ├── "allow"   → guarda la imagen
        ├── "review"  → cuarentena + cola de moderación humana
        └── "block"   → 4xx al usuario + registra {request_id, sha256, risk_score}
```

---

## Política de moderación

El modelo no dice "es NSFW": dice **qué partes del cuerpo ve**. Cada etiqueta tiene una severidad y
un peso, y eso es lo que decide el veredicto.

| Severidad | Peso | Etiquetas |
|---|---|---|
| `explicit` | 0.85 – 1.00 | `FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED`, `ANUS_EXPOSED`, `BUTTOCKS_EXPOSED` (0.90), `FEMALE_BREAST_EXPOSED` (0.85) |
| `suggestive` | 0.05 – 0.55 | `FEMALE_GENITALIA_COVERED` (0.55), `BUTTOCKS_COVERED`, `FEMALE_BREAST_COVERED`, `MALE_BREAST_EXPOSED` (0.15), `BELLY_EXPOSED` (0.10), `ARMPITS_EXPOSED` (0.05) |
| `neutral` | 0.00 | `FACE_FEMALE`, `FACE_MALE`, `FEET_*`, `BELLY_COVERED`, `ARMPITS_COVERED` |

Un torso masculino en la playa pesa **0.15**; unos genitales expuestos pesan **1.00**. Ahí está la
diferencia entre un filtro útil y uno que bloquea a cualquiera que suba una foto de vacaciones.

Cada detección aporta `p = peso × confianza × factor_de_área` (una caja minúscula al fondo pesa
menos que un primer plano). Las aportaciones se combinan con OR ruidoso (`1 − Π(1 − pᵢ)`) y el
riesgo final es `max(explícito, factor_sugerente × sugerente)`.

Perfiles:

| Perfil | Bloqueo ≥ | Revisión ≥ | Peso sugerente | Cuándo usarlo |
|---|---|---|---|---|
| `strict` | 0.30 | 0.12 | 1.00 | Apps infantiles o escolares: bloquea incluso ropa interior y bikinis |
| `balanced` *(default)* | 0.55 | 0.28 | 0.75 | Desnudos explícitos fuera; lo ambiguo a revisión humana |
| `lenient` | 0.80 | 0.50 | 0.45 | Contextos artísticos o médicos donde un falso positivo cuesta caro |

Ajuste fino por etiqueta:

```bash
# Tolerar torso masculino y ser estricto con glúteos:
NID_LABEL_WEIGHTS='{"MALE_BREAST_EXPOSED":0.0,"BUTTOCKS_EXPOSED":1.0}'
```

Los tres veredictos existen a propósito: el intervalo `review` es donde un humano decide los casos
dudosos, mucho mejor que forzar un sí/no.

---

## Configuración

Variables de entorno con prefijo `NID_` (o un archivo `.env`; copia `.env.example`). Las variables
reales del entorno tienen prioridad sobre el `.env`, y **se validan al arrancar**: un umbral de
revisión mayor que el de bloqueo o un JSON inválido fallan rápido en vez de debilitar la moderación
en silencio.

| Variable | Por defecto | Descripción |
|---|---|---|
| `NID_HOST` / `NID_PORT` | `0.0.0.0` / `8000` | Dónde escucha el servicio |
| `NID_ENGINE` | `nudenet` | `nudenet` (modelo real) · `mock` (siempre `allow`, **solo pruebas**) · `disabled` |
| `NID_STRICTNESS` | `balanced` | Perfil de moderación por defecto |
| `NID_MIN_CONFIDENCE` | `0.30` | Confianza mínima para puntuar una detección |
| `NID_LABEL_WEIGHTS` | *(vacío)* | JSON con pesos por etiqueta (0–1) |
| `NID_BLOCK_THRESHOLD` / `NID_REVIEW_THRESHOLD` | *(perfil)* | Sobrescritura global de umbrales |
| `NID_ALLOW_REQUEST_OVERRIDES` | `true` | Permite cambiar perfil/umbrales por petición |
| `NID_MAX_CONTENT_LENGTH` | `10485760` | Bytes máximos por imagen |
| `NID_MAX_IMAGE_PIXELS` | `40000000` | Tope de píxeles (anti bomba de descompresión) |
| `NID_MAX_ANALYSIS_PIXELS` | `4000000` | Por encima, se reduce antes de inferir |
| `NID_MAX_FRAMES` | `3` | Fotogramas muestreados en imágenes animadas |
| `NID_ALLOWED_MIME_TYPES` | jpeg,png,webp,gif,bmp,tiff | Formatos aceptados |
| `NID_API_KEYS` | *(vacío)* | Claves separadas por comas; en cuanto defines una, se exige |
| `NID_RATE_LIMIT_ENABLED` | `false` | Límite de peticiones por proceso |
| `NID_FETCH_ENABLED` | `true` | Permite el análisis por URL |
| `NID_FETCH_ALLOW_PRIVATE_NETWORKS` | `false` | **Solo pruebas**: desactiva la protección SSRF |
| `NID_FETCH_ALLOWED_HOSTS` | *(vacío)* | Allowlist de hosts para URLs |
| `NID_FETCH_TIMEOUT_SECONDS` | `8` | Timeout por salto de redirección |
| `NID_BATCH_MAX_ITEMS` | `10` | Máximo de imágenes por lote |
| `NID_MAX_CONCURRENT_INFERENCES` | `2` | Semáforo de inferencias simultáneas por proceso |
| `NID_CACHE_ENABLED` / `NID_CACHE_TTL_SECONDS` | `true` / `3600` | Caché de detecciones por SHA-256 |
| `NID_LOG_JSON` | `true` | Logs JSON (agregadores) o texto plano (local) |
| `NID_TRUSTED_PROXY_COUNT` | `0` | Proxies delante de la app para interpretar `X-Forwarded-For` |

Lista completa y comentada en [`.env.example`](.env.example).

**Dimensionado**: la inferencia es *CPU-bound* y cada worker de gunicorn carga su propia sesión ONNX
(≈110 MB RSS). Mejor pocos workers con varios hilos (`NID_GUNICORN_WORKERS=2`,
`NID_GUNICORN_THREADS=4`) que muchos workers, y limita la concurrencia real con
`NID_MAX_CONCURRENT_INFERENCES`. Rendimiento medido en 2 vCPU sin GPU: carga del modelo ≈ 0.14 s,
inferencia ≈ 26–38 ms por fotograma, duplicados en caché < 1 ms.

---

## Seguridad

Este servicio recibe bytes y URLs de usuarios anónimos, así que la superficie de ataque importa tanto
como el modelo.

**Al aceptar URLs (anti-SSRF)** — sin esto, tu API sería un proxy abierto:
- Solo `http`/`https`; se validan host, puerto, credenciales embebidas y caracteres de control.
- Resolución DNS previa: **todas** las IPs devueltas deben ser públicas. Se bloquean loopback,
  link-local (**incluido `169.254.169.254`**, el *metadata* de la nube), privadas, CGNAT, multicast,
  reservadas e IPv4 mapeada en IPv6 (`::ffff:127.0.0.1`).
- Las redirecciones se siguen manualmente, revalidando cada salto.
- Tope por `Content-Length` **y** por lectura en streaming (un servidor que miente no gana).
- Tras conectar se verifica la IP real del socket, lo que mitiga *DNS rebinding*.

**Al aceptar imágenes**:
- El formato real se olfatea por *magic bytes*; nunca se confía en la extensión ni en `Content-Type`.
- Doble tope: bytes y píxeles (una bomba de descompresión de 50 000×50 000 se rechaza sin decodificar).
- Las imágenes truncadas o corruptas se rechazan en lugar de analizarse a medias.
- **HEIC/AVIF se detectan y se rechazan con un mensaje claro**: OpenCV no los decodifica y fingir lo
  contrario dejaría pasar una foto explícita como "segura".
- Se aplica la orientación EXIF y se muestrean varios fotogramas de GIF/WebP animados (un GIF no
  puede esconder frames explícitos tras uno inocente).

**Resto**:
- Autenticación opcional por `X-API-Key` / `Bearer`, comparada en tiempo constante.
- Rate limiting opcional por identidad, con cabeceras `X-RateLimit-*` y `Retry-After`.
- `X-Forwarded-For` solo se interpreta con `NID_TRUSTED_PROXY_COUNT` > 0.
- Cabeceras `X-Content-Type-Options: nosniff` y `Cache-Control: no-store` (las respuestas pueden
  llevar una imagen censurada en base64).
- **Privacidad**: las imágenes no se escriben a disco ni se registran. En los logs solo aparecen
  `sha256`, veredicto, riesgo y etiquetas; las URLs se registran sin *query string* para no filtrar
  tokens firmados.
- Los errores nunca filtran rutas internas ni trazas al cliente.

---

## Estructura

```
app/
├── __init__.py          # create_app(): fábrica, middleware, cabeceras, logging
├── config.py            # Settings inmutables desde env, validadas al arrancar
├── errors.py            # Excepciones tipadas → códigos estables + HTTP
├── logging_conf.py      # Logs JSON con request_id; nunca registra imágenes
├── security.py          # API keys, IP de cliente, rate limiting
├── service.py           # Orquestación: entrada → imagen → motor → política → respuesta
├── api/
│   ├── routes.py        # Endpoints REST
│   └── errors.py        # Manejadores que mantienen el contrato JSON
└── core/
    ├── labels.py        # Catálogo de etiquetas → severidad + peso
    ├── imaging.py       # Validación y normalización segura de imágenes
    ├── fetcher.py       # Descarga remota con protección anti-SSRF
    ├── engine.py        # Motores de detección (NudeNet ONNX, mock, disabled)
    ├── policy.py        # Detecciones → riesgo → veredicto explicable
    └── cache.py         # LRU + TTL thread-safe
```

Cambiar el modelo significa tocar **una clase** (`DetectionEngine`): la capa HTTP y la política no se
enteran.

---

## Limitaciones

- **Ningún clasificador visual es infalible.** Espera **falsos positivos** en playa/piscina, arte
  clásico, lactancia, contenido médico o deportivo, y **falsos negativos** con desnudos parciales,
  ilustraciones, imágenes generadas por IA, muy pequeñas o muy comprimidas.
- El modelo se entrenó con fotografía real: el contenido generado por IA y las ilustraciones
  explícitas se detectan peor.
- **No detecta** texto, contexto, edad de las personas ni otros riesgos (violencia, drogas,
  documentos de identidad). Es solo desnudez.
- La caché y el rate limit son **por proceso**; con varias réplicas necesitarás Redis o hacerlo en tu
  pasarela.
- Recomendación: arranca **una o dos semanas en modo observación** (registra veredictos sin
  bloquear), mide tus falsos positivos reales y luego ajusta `NID_STRICTNESS` / `NID_LABEL_WEIGHTS`.
  Guarda `risk_score` y `sha256` para poder re-evaluar tu histórico al cambiar umbrales.
- **Aviso legal**: un detector de desnudos trata datos potencialmente sensibles. Revisa tu base legal
  (consentimiento, interés legítimo) y tu política de privacidad antes de procesar imágenes de
  usuarios, sobre todo bajo GDPR/LGPD.

---

## Licencia

MIT — ver [LICENSE](LICENSE). El modelo `320n.onnx` se distribuye a través del paquete
[`nudenet`](https://pypi.org/project/nudenet/) con su propia licencia.
