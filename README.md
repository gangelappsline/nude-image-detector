# nude-image-detector

API REST en **Python + Flask** que analiza una imagen (subida multipart, URL remota o base64) y
devuelve un **veredicto de moderación** —`block` / `review` / `allow`— junto con el riesgo de
desnudez detectado. Está pensada para interponerse **antes** de que las imágenes de tus usuarios
lleguen a tu almacenamiento (S3, disco, base de datos).

```
POST /v1/analyze  ──►  {"verdict":"block","nsfw":true,"risk_score":0.93,
                        "reasons":["FEMALE_BREAST_EXPOSED: pecho femenino expuesto (confianza 0.94, área 6.1%, peso 0.85)"],
                        "detections":[{"label":"FEMALE_BREAST_EXPOSED","confidence":0.94,"box":[412,180,733,540], ...}]}
```

---

## Índice

- [Qué hace](#qué-hace)
- [Cómo funciona](#cómo-funciona)
- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Referencia de la API](#referencia-de-la-api)
- [Política de moderación](#política-de-moderación)
- [Configuración](#configuración)
- [Seguridad](#seguridad)
- [Despliegue](#despliegue)
- [Integración recomendada](#integración-recomendada)
- [Pruebas](#pruebas)
- [Limitaciones](#limitaciones)
- [Arquitectura del código](#arquitectura-del-código)

---

## Qué hace

| Capacidad | Detalle |
|---|---|
| **Entradas** | Archivo multipart, URL pública, base64 (admite *data URLs*) o cuerpo binario crudo |
| **Salida** | Veredicto, `risk_score` 0–1, puntuaciones por severidad, motivos legibles, cajas de detección y metadatos de la imagen |
| **Modelo** | YOLOv8-nano de NudeNet (`320n.onnx`, 12 MB) vía ONNX Runtime — **viene dentro del wheel de PyPI**, no se descarga nada en runtime |
| **Lotes** | Hasta `NID_BATCH_MAX_ITEMS` imágenes por petición; un fallo individual no aborta el lote |
| **GIF/animados** | Muestrea varios fotogramas: un GIF no puede esconder frames explícitos tras uno inocente |
| **Censura** | `?censor=true` devuelve una copia pixelada y difuminada en base64 |
| **Caché** | Por SHA-256 de la imagen: reintentos y duplicados no vuelven a ejecutar el modelo |
| **Docs vivas** | `GET /` sirve una página con referencia + *playground* interactivo (sin CDNs) y `GET /openapi.json` la spec |

**Rendimiento medido** (2 vCPU, sin GPU): carga del modelo ≈ 0.15 s, inferencia ≈ 30 ms por fotograma,
≈ 110 MB de RSS por worker.

---

## Cómo funciona

```
                    ┌──────────────────────────────────────────────────────────┐
  multipart ───────►│  app/api/routes.py     contrato HTTP, parámetros, lotes  │
  {"url": ...} ────►│                                                          │
  {"image_base64"}─►└───────────────┬──────────────────────────────────────────┘
  cuerpo binario ──►                │
                                    ▼
              ┌──────────────────────────────────────┐   URL   ┌────────────────────────┐
              │        app/service.py                │◄────────│  app/core/fetcher.py   │
              │  orquestación + caché por SHA-256    │         │  validación anti-SSRF  │
              └───────┬──────────────────────┬───────┘         └────────────────────────┘
                      │                      │
                      ▼                      ▼
      ┌────────────────────────┐   ┌────────────────────────┐
      │  app/core/imaging.py   │   │  app/core/engine.py    │
      │  validación, EXIF,     │   │  NudeNet ONNX +        │
      │  frames, normalización │   │  semáforo de CPU       │
      └────────────────────────┘   └───────────┬────────────┘
                                               ▼
                                  ┌────────────────────────┐
                                  │  app/core/policy.py    │
                                  │  detecciones → riesgo  │
                                  │  → allow/review/block  │
                                  └────────────────────────┘
```

El modelo **no** decide; solo dice qué partes del cuerpo ve y con qué confianza.
La capa de política (`app/core/policy.py` + `app/core/labels.py`) convierte eso en una decisión
explicable y ajustable para tu producto.

---

## Instalación

Requisitos: **Python 3.10+** (probado en 3.11). No hace falta GPU ni descargar pesos.

```bash
git clone https://github.com/gangelappsline/nude-image-detector.git
cd nude-image-detector

make install        # crea .venv e instala dependencias
# o manualmente:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

make serve          # gunicorn en http://0.0.0.0:8000
```

Comprueba que todo está vivo:

```bash
curl -s http://127.0.0.1:8000/ready | jq
# {"ready": true, "engine": "nudenet", "model": "320n@320", ...}
```

Abre `http://127.0.0.1:8000/` en el navegador para la documentación interactiva.

---

## Uso rápido

```bash
HOST=http://127.0.0.1:8000

# 1) Subir un archivo local
curl -s -X POST "$HOST/v1/analyze" -F "file=@foto.jpg" | jq '{verdict, nsfw, risk_score, reasons}'

# 2) Analizar una URL remota
curl -s -X POST "$HOST/v1/analyze" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://ejemplo.com/foto.jpg"}' | jq

# 3) Imagen en base64
curl -s -X POST "$HOST/v1/analyze" \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$(base64 -w0 foto.jpg)\"}" | jq

# 4) Cuerpo binario crudo
curl -s -X POST "$HOST/v1/analyze" -H 'Content-Type: image/jpeg' --data-binary @foto.jpg | jq

# 5) Perfil estricto + copia censurada
curl -s -X POST "$HOST/v1/analyze?strictness=strict&censor=true" -F "file=@foto.jpg" | jq

# 6) Lote de URLs
curl -s -X POST "$HOST/v1/analyze/batch" \
  -H 'Content-Type: application/json' \
  -d '{"urls":["https://a.com/1.jpg","https://b.com/2.jpg"]}' | jq '.summary'

# 7) Con autenticación
curl -s -X POST "$HOST/v1/analyze" -H "X-API-Key: $NID_API_KEY" -F "file=@foto.jpg" | jq
```

Desde Python:

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

---

## Referencia de la API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/v1/analyze` | Analiza una imagen (upload, URL, base64 o binario) |
| `POST` | `/v1/analyze/batch` | Analiza hasta `NID_BATCH_MAX_ITEMS` imágenes |
| `GET` | `/v1/info` | Modelo, perfiles y límites del despliegue |
| `GET` | `/v1/labels` | Catálogo de etiquetas con severidad y peso |
| `GET` | `/health` | *Liveness probe* (pública) |
| `GET` | `/ready` | *Readiness probe*: 503 si el modelo no cargó (pública) |
| `GET` | `/openapi.json` | Especificación OpenAPI 3.0 |
| `DELETE` | `/v1/cache` | Vacía la caché (requiere `NID_API_KEYS`) |
| `GET` | `/` | Documentación interactiva + playground |

### Parámetros de `POST /v1/analyze`

Aceptados como *query string*, campo de formulario o clave del JSON:

| Parámetro | Tipo | Descripción |
|---|---|---|
| `strictness` | `strict` \| `balanced` \| `lenient` | Perfil de estrictez |
| `block_threshold` | 0–1 | Sobrescribe el umbral de bloqueo |
| `review_threshold` | 0–1 | Sobrescribe el umbral de revisión |
| `min_confidence` | 0–1 | Confianza mínima para que una detección puntúe |
| `censor` | bool | Incluye una copia censurada en base64 si no pasa el filtro |
| `no_detections` | bool | Omite la lista de detecciones (respuesta más ligera) |
| `reject_on_block` | bool | Devuelve **HTTP 422** en vez de 200 cuando el veredicto es `block` |

> Las sobrescrituras por petición se pueden desactivar con `NID_ALLOW_REQUEST_OVERRIDES=false`,
> útil cuando quieres que la política la fije únicamente el despliegue.

### Respuesta

```jsonc
{
  "request_id": "9f1c0a5b6d7e8f90",
  "verdict": "block",                    // block | review | allow
  "verdict_description_es": "Rechaza la subida: se detectó desnudez con confianza suficiente.",
  "nsfw": true,                          // true cuando verdict == "block"
  "risk_score": 0.9288,                  // 0..1, agregado de todos los hallazgos
  "scores": { "explicit": 0.9288, "suggestive": 0.0 },
  "reasons": [
    "FEMALE_BREAST_EXPOSED: pecho femenino expuesto (confianza 0.94, área 6.1%, peso 0.85)"
  ],
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
      "severity": "explicit",
      "weight": 0.85,
      "box": [412, 180, 733, 540],       // x1, y1, x2, y2 en píxeles
      "box_xywh": [412, 180, 321, 360],
      "area_ratio": 0.0611,              // fracción del fotograma que ocupa
      "frame_index": 0,
      "counted": true,                   // false si quedó bajo min_confidence
      "description_es": "Pecho femenino expuesto (pezón/areola)"
    }
  ],
  "policy": { "profile": "balanced", "block_threshold": 0.55, "review_threshold": 0.28, "...": "..." },
  "image": {
    "format": "jpeg", "mime_type": "image/jpeg", "width": 1200, "height": 800,
    "source_width": 4000, "source_height": 2667, "bytes": 481203,
    "sha256": "a8ce5da1...", "animated": false, "frames_total": 1,
    "frames_analysed": 1, "downscaled": true, "orientation_applied": true, "mime_mismatch": false
  },
  "source": { "type": "upload", "filename": "foto.jpg", "bytes": 481203 },
  "model": { "engine": "nudenet", "model": "320n@320", "version": "3.4.2", "ready": true },
  "cached": false,
  "elapsed_ms": 41.7
}
```

**Un análisis correcto siempre devuelve `200 OK`**, incluso si el veredicto es `block`: la API hizo
su trabajo. Usa `?reject_on_block=true` si tu pasarela prefiere un error HTTP para subir rechazadas.

### Errores

Todas las respuestas de error comparten forma:

```json
{ "request_id": "9f1c0a5b", "error": { "code": "blocked_destination", "message": "...", "status": 403, "details": {} } }
```

| `code` | HTTP | Cuándo ocurre |
|---|---|---|
| `missing_image` | 400 | No llegó ninguna imagen |
| `bad_request` | 400 | Parámetros o JSON inválidos |
| `invalid_url` | 400 | URL mal formada, esquema no http(s), credenciales embebidas |
| `too_many_items` | 400 | El lote supera `NID_BATCH_MAX_ITEMS` |
| `unauthorized` | 401 | Falta o es inválida `X-API-Key` |
| `forbidden` | 403 | Operación no permitida (p. ej. URL fetching desactivado) |
| `blocked_destination` | 403 | La URL apunta a una red privada o protegida (anti-SSRF) |
| `payload_too_large` | 413 | Se superó `NID_MAX_CONTENT_LENGTH` o `NID_FETCH_MAX_BYTES` |
| `unsupported_media_type` | 415 | Formato no permitido (p. ej. HEIC/AVIF) |
| `unprocessable_image` | 422 | El archivo no se pudo decodificar como imagen |
| `image_too_large` | 422 | Demasiados píxeles (posible bomba de descompresión) |
| `rate_limited` | 429 | Límite de peticiones superado (incluye `Retry-After`) |
| `upstream_error` | 502 | El servidor remoto falló o devolvió un estado inesperado |
| `upstream_timeout` | 504 | El servidor remoto tardó demasiado |
| `inference_error` | 500 | El modelo falló al procesar la imagen |
| `model_unavailable` | 503 | El motor de detección no está cargado |

Programa contra `error.code`, nunca contra el texto del mensaje.

### Lotes

```jsonc
POST /v1/analyze/batch
{ "urls": ["https://a.com/1.jpg", "https://b.com/2.jpg"] }
```

```jsonc
{
  "request_id": "…",
  "count": 2,
  "summary": {
    "total": 2, "succeeded": 1, "failed": 1,
    "verdicts": { "allow": 1, "review": 0, "block": 0, "error": 1 },
    "highest_risk_score": 0.04, "any_blocked": false,
    "policy": { "profile": "balanced", "block_threshold": 0.55 }
  },
  "results": [
    { "index": 0, "source": { "type": "url", "host": "a.com" }, "ok": true,  "result": { "verdict": "allow", "…": "…" } },
    { "index": 1, "source": { "type": "url", "host": "b.com" }, "ok": false, "error": { "code": "upstream_error", "status": 502 } }
  ],
  "elapsed_ms": 180.4
}
```

También acepta `multipart/form-data` con varias partes (`files`), o
`{"items":[{"url":…},{"image_base64":…}]}` para mezclar orígenes.

---

## Política de moderación

### Severidades

Cada etiqueta del modelo se clasifica en un nivel con un peso por defecto:

| Severidad | Peso | Etiquetas |
|---|---|---|
| `explicit` | 0.85 – 1.00 | `FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED`, `ANUS_EXPOSED`, `BUTTOCKS_EXPOSED` (0.90), `FEMALE_BREAST_EXPOSED` (0.85) |
| `suggestive` | 0.05 – 0.55 | `FEMALE_GENITALIA_COVERED` (0.55), `BUTTOCKS_COVERED`, `FEMALE_BREAST_COVERED`, `MALE_BREAST_EXPOSED` (0.15), `BELLY_EXPOSED` (0.10), `ARMPITS_EXPOSED` (0.05), … |
| `neutral` | 0.00 | `FACE_FEMALE`, `FACE_MALE`, `FEET_*`, `BELLY_COVERED`, `ARMPITS_COVERED` |

Un torso masculino en la playa pesa **0.15**; unos genitales expuestos pesan **1.00**. Ahí está la
diferencia entre un filtro útil y uno que bloquea a cualquiera que suba una foto de vacaciones.

### Puntuación

Cada detección aporta:

```
p = peso(etiqueta) × confianza × factor_de_área(caja)
```

- `factor_de_área` ∈ [`area_floor`, 1] = [0.75, 1]: una caja minúscula al fondo pesa menos que un
  primer plano; satura al cubrir el 1 % del fotograma.
- Las aportaciones se combinan con **OR ruidoso** (`1 − Π(1 − pᵢ)`), la agregación correcta para
  "basta uno de estos hallazgos": satura hacia 1 en lugar de desbordarse como una suma.
- Se calculan dos puntuaciones, `explicit` y `suggestive`, y el riesgo final es:

```
risk = max(explicit, suggestive_factor × suggestive)
       ↑ con un suelo para un único hallazgo explícito fuerte (0.9 × peso × confianza)
```

### Perfiles

| Perfil | Bloqueo ≥ | Revisión ≥ | Peso sugerente | Cuándo usarlo |
|---|---|---|---|---|
| `strict` | 0.30 | 0.12 | 1.00 | Apps infantiles o escolares: bloquea incluso ropa interior y bikinis |
| `balanced` *(default)* | 0.55 | 0.28 | 0.75 | Desnudos explícitos fuera; contenido ambiguo a revisión humana |
| `lenient` | 0.80 | 0.50 | 0.45 | Contextos artísticos o médicos donde un falso positivo cuesta caro |

### Ajuste fino por etiqueta

```bash
# Tolerar torso masculino y ser estricto con glúteos:
NID_LABEL_WEIGHTS='{"MALE_BREAST_EXPOSED":0.0,"BUTTOCKS_EXPOSED":1.0}'
```

Los tres veredictos existen a propósito: el intervalo `review` es donde un humano decide los casos
dudosos. Conectarlo a una cola de moderación es mucho mejor que forzar un sí/no.

---

## Configuración

Todo se configura con variables de entorno (`NID_`), o en un archivo `.env`
(copia `.env.example`). Las variables reales del entorno tienen prioridad sobre el `.env`.
Lista completa y comentada en [`.env.example`](.env.example); las más relevantes:

| Variable | Por defecto | Descripción |
|---|---|---|
| `NID_ENGINE` | `nudenet` | `nudenet` (modelo real) · `mock` (siempre `allow`, **solo pruebas**) · `disabled` |
| `NID_STRICTNESS` | `balanced` | Perfil de moderación por defecto |
| `NID_MIN_CONFIDENCE` | `0.30` | Confianza mínima para puntuar una detección |
| `NID_LABEL_WEIGHTS` | *(vacío)* | JSON con pesos por etiqueta (0–1) |
| `NID_MAX_CONTENT_LENGTH` | `10485760` | Bytes máximos por imagen |
| `NID_MAX_IMAGE_PIXELS` | `40000000` | Tope de píxeles (anti bomba de descompresión) |
| `NID_MAX_ANALYSIS_PIXELS` | `4000000` | Por encima, se reduce antes de inferir |
| `NID_MAX_FRAMES` | `3` | Fotogramas muestreados en imágenes animadas |
| `NID_API_KEYS` | *(vacío)* | Claves separadas por comas; en cuanto defines una, se exige |
| `NID_RATE_LIMIT_ENABLED` | `false` | Límite de peticiones por proceso |
| `NID_FETCH_ENABLED` | `true` | Permite el análisis por URL |
| `NID_FETCH_ALLOW_PRIVATE_NETWORKS` | `false` | **Solo pruebas**: desactiva la protección SSRF |
| `NID_FETCH_ALLOWED_HOSTS` | *(vacío)* | Allowlist de hosts para URLs |
| `NID_BATCH_MAX_ITEMS` | `10` | Máximo de imágenes por lote |
| `NID_MAX_CONCURRENT_INFERENCES` | `2` | Semáforo de inferencias simultáneas por proceso |
| `NID_CACHE_ENABLED` | `true` | Caché de detecciones por SHA-256 |
| `NID_LOG_JSON` | `true` | Logs JSON (agregadores) o texto plano (local) |

La configuración se **valida al arrancar**: un umbral de revisión mayor que el de bloqueo, un JSON
inválido en `NID_LABEL_WEIGHTS` o un perfil inexistente fallan rápido en vez de debilitar la
moderación en silencio.

---

## Seguridad

Este servicio recibe bytes y URLs de usuarios anónimos: la superficie de ataque importa más que el
modelo.

**Validación de imágenes** (`app/core/imaging.py`)
- El formato real se olfatea por **magic bytes**; nunca se confía en la extensión ni en `Content-Type`.
- Doble tope: **bytes** y **píxeles** (una bomba de descompresión de 50 000×50 000 se rechaza sin
  decodificar).
- Las imágenes truncadas o corruptas se rechazan en lugar de analizarse a medias.
- **HEIC/AVIF se detectan y se rechazan con un mensaje accionable**: OpenCV no los decodifica y
  fingir lo contrario dejaría pasar una foto explícita como "segura".
- Se aplica la **orientación EXIF** y se muestrean **varios fotogramas** de GIF/WebP animados.

**Protección anti-SSRF** (`app/core/fetcher.py`)
1. Solo `http`/`https`; puertos, hosts y credenciales embebidas validados; caracteres de control
   rechazados (evita inyección de cabeceras).
2. Resolución DNS previa: **todas** las direcciones devueltas deben ser públicas. Se bloquean
   loopback, link-local (incluido `169.254.169.254`, el *metadata* de la nube), privadas, CGNAT,
   multicast, reservadas y las IPv4 mapeadas en IPv6 (`::ffff:127.0.0.1`).
3. Las redirecciones se siguen **manualmente**, revalidando cada salto desde cero.
4. Tope por `Content-Length` **y** por lectura en streaming (un servidor que miente no gana).
5. Tras conectar se verifica la **IP real del socket**: si el DNS hizo *rebinding* hacia una red
   privada, se descarta el cuerpo.
6. `trust_env = False`: nunca se usan proxies ni credenciales del entorno hacia terceros.

**Otros**
- Autenticación opcional por `X-API-Key` / `Bearer`, comparada en **tiempo constante**.
- Rate limiting por identidad (clave o IP), con cabeceras `X-RateLimit-*` y `Retry-After`.
- `X-Forwarded-For` solo se interpreta con `NID_TRUSTED_PROXY_COUNT` > 0 (y tomando los saltos
  correctos desde la derecha).
- Cabeceras de seguridad: `X-Content-Type-Options`, `Referrer-Policy: no-referrer`, CSP, y
  `Cache-Control: no-store` en las respuestas de análisis (pueden llevar imagen censurada en base64).
- **Privacidad**: las imágenes no se escriben a disco ni se registran. En los logs solo aparecen
  `sha256`, veredicto, riesgo y etiquetas; las URLs se registran **sin query string** para no filtrar
  tokens firmados.
- Los errores nunca filtran rutas internas ni trazas al cliente.
- Contenedor: usuario sin privilegios, `read_only`, `no-new-privileges`, límites de CPU/memoria y
  *healthcheck*.

---

## Despliegue

### Docker

```bash
make docker                                     # construye la imagen
docker run --rm -p 8000:8000 \
  -e NID_STRICTNESS=balanced \
  -e NID_API_KEYS="mi-clave-segura" \
  nude-image-detector:latest

# o con compose (lee .env):
docker compose up --build
```

La imagen es autocontenida: los pesos van dentro del wheel de `nudenet`, así que **no hay descargas
en runtime** y puede arrancar sin salida a internet.

### Gunicorn / systemd

```bash
NID_GUNICORN_WORKERS=2 NID_GUNICORN_THREADS=4 \
  .venv/bin/gunicorn -c gunicorn.conf.py wsgi:app
```

Regla de dimensionado: la inferencia es *CPU-bound* y cada worker carga su propia sesión ONNX
(≈110 MB RSS). Mejor **pocos workers con varios hilos** que muchos workers, y limita la concurrencia
real con `NID_MAX_CONCURRENT_INFERENCES`. `preload_app` está desactivado a propósito: bifurcar
*después* de crear la sesión ONNX no es seguro.

### Kubernetes

```yaml
livenessProbe:  { httpGet: { path: /health, port: 8000 } }
readinessProbe: { httpGet: { path: /ready,  port: 8000 } }   # 503 hasta que el modelo cargue
resources:      { requests: { cpu: "500m", memory: "256Mi" }, limits: { cpu: "2", memory: "1Gi" } }
```

### Detrás de un reverse proxy

Establece `NID_TRUSTED_PROXY_COUNT=1` (o el número de proxies reales) para que el rate limiting y
los logs usen la IP correcta del cliente.

---

## Integración recomendada

```
usuario sube imagen
        │
        ▼
POST /v1/analyze  (antes de escribir en S3 / tu BD)
        │
        ├── verdict == "allow"   → guarda la imagen
        ├── verdict == "review"  → cuarentena + cola de moderación humana
        └── verdict == "block"   → 4xx al usuario + registra {request_id, sha256, risk_score}
```

Consejos de producción:

1. **Analiza antes de persistir.** Si ya guardaste la imagen, el daño está hecho.
2. **No dependas solo del veredicto**: guarda `risk_score` y `sha256`. Te permitirán re-evaluar
   todo tu histórico cuando cambies los umbrales, sin volver a analizar nada.
3. **Empieza en modo observación** (registra veredictos sin bloquear) una o dos semanas, mide tus
   falsos positivos reales y luego ajusta `NID_STRICTNESS` / `NID_LABEL_WEIGHTS`.
4. **Reutiliza el `sha256`** como clave en tu propio almacenamiento: la misma foto repetida no
   debería volver a pagar inferencia.
5. **Timeout del cliente**: 30 s es holgado; una imagen normal se resuelve en <200 ms.
6. **Aviso legal**: un detector de desnudos trata datos potencialmente sensibles. Revisa tu base
   legal (consentimiento, interés legítimo) y tu política de privacidad antes de procesar imágenes
   de usuarios, especialmente en GDPR/LGPD.

---

## Pruebas

```bash
make test     # suite completa (249 pruebas, incluye el modelo ONNX real)
make fast     # sin cargar el modelo real
make lint     # ruff
```

La suite no necesita internet ni fotografías reales:

- `tests/test_policy.py` — matemática de puntuación, perfiles, sobrescrituras de peso.
- `tests/test_imaging.py` — magic bytes, bombas de descompresión, EXIF, GIF animado, modos exóticos
  (paleta, CMYK, 16 bits), censura.
- `tests/test_fetcher.py` — **~90 casos de SSRF**: loopback, metadata de la nube, CGNAT, IPv6,
  IPv4 mapeado, *rebinding* por DNS, bucles de redirección, topes de tamaño.
- `tests/test_api.py` — contrato HTTP, modos de entrada, auth, rate limit, lotes, cabeceras.
- `tests/test_engine.py` — normalización de cajas, semáforo de concurrencia, manejo de fallos.
- `tests/test_model_e2e.py` — modelo real contra imágenes sintéticas (marcadas `slow`).

Comprobación rápida contra un servidor ya arrancado:

```bash
make smoke                                  # o: python scripts/smoke_test.py --base-url http://localhost:8000
```

---

## Limitaciones

Sé honesto con esto antes de conectarlo a producción:

- **Ningún clasificador visual es infalible.** Espera **falsos positivos** en playa/piscina, arte
  clásico, lactancia, contenido médico o deportivo; y **falsos negativos** con desnudos parciales,
  ilustraciones/dibujos, imágenes muy pequeñas o muy comprimidas, y oclusiones.
- El modelo se entrenó principalmente con **fotografía real**: el contenido generado por IA o las
  ilustraciones explícitas se detectan peor.
- **No detecta** texto, contexto, edad de las personas ni otros riesgos (violencia, drogas,
  documentos de identidad). Es solo desnudez.
- El análisis es por fotograma a 320×320: los detalles pequeños pueden perderse.
- La caché y el rate limit son **por proceso**; con varias réplicas necesitarás Redis o hacerlo en
  tu pasarela.
- Por eso la API devuelve **tres veredictos y no un sí/no**: el rango `review` existe para que un
  humano decida los casos dudosos.

---

## Arquitectura del código

```
app/
├── __init__.py          # create_app(): fábrica, middleware, cabeceras, logging de acceso
├── config.py            # Settings inmutables desde env, con validación al arrancar
├── errors.py            # Excepciones tipadas → códigos estables + HTTP
├── logging_conf.py      # Logs JSON con request_id; nunca registra imágenes
├── security.py          # API keys, IP de cliente, rate limiting
├── service.py           # Orquestación: entrada → imagen → motor → política → respuesta
├── openapi.py           # Spec OpenAPI 3.0 escrita a mano
├── ui.py                # Blueprint de documentación + playground
├── templates/index.html # Docs interactivas autocontenidas (sin CDNs)
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

Cambiar el modelo significa tocar **una clase** (`DetectionEngine`); la capa HTTP y la política no
se enteran.

---

## Licencia

MIT — ver [LICENSE](LICENSE). El modelo `320n.onnx` se distribuye a través del paquete
[`nudenet`](https://pypi.org/project/nudenet/) con su propia licencia.
