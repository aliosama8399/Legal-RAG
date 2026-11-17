# Data & SaaS API

The API is organized into layered modules under `src/law_api/`:

- `src/law_api/auth/`: Multi-tenant organizations, PBKDF2/JWT/prefixed API keys (`lrag_live_...`), RBAC dependencies, SQLite WAL state store.
- `src/law_api/routes/`: `saas_routes.py`, `data_routes.py`, `rag_routes.py`, `health_routes.py`.
- `src/law_api/jobs/ingestion_worker.py`: Async `202 Accepted` ingestion pipeline with `Idempotency-Key` deduplication.
- `src/law_api/services.py`: Hybrid RAG query service, conversation context management, usage metering, and live evaluation.
- `src/law_api/stores/`: Pluggable `vectordb` (`qdrant-local`, `qdrant`, `postgresql`), `embeddings`, `rerankers`, and `llm` (`openai`, `gemini`, `anthropic`/`claude`, `openrouter`, `groq`, `deepseek`, `vllm`, `ollama`, `local`, `mock`).
- `src/law_api/schemas.py`, `middleware.py`, `exceptions.py`: Pydantic contracts, sliding-window rate limiter, `X-Request-ID`, RFC 9457 error formatting.

Configuration is split into:

- `config/providers.yaml`: non-secret provider and model choices.
- `.env`: local secrets and environment-specific overrides. Copy `.env.example`
  to `.env`; `.env` is ignored by git.

Environment variables override values from `config/providers.yaml`.

## Run

```powershell
uv run uvicorn law_api.main:app --reload --host 127.0.0.1 --port 8000
```

The interactive documentation is available at `http://127.0.0.1:8000/docs`.

## Upload and chunk

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents/upload `
  -F "file=@src/data/raw/egyptian_civil_code.pdf"
```

The upload endpoint only:

1. Validates the PDF upload and size.
2. Stores the original file under `storage/uploads`.
3. Extracts bilingual article records.
4. Creates citation-preserving chunks.
5. Saves pending chunks under `storage/pending`.

It does not load an embedding model or start an MLflow run.

## Embed and store

Use the returned `document_id`:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents/1/embed
```

This endpoint loads the selected embedding model, embeds the pending chunks,
stores vectors, and creates the MLflow run.

## Provider and embedding comparisons

PostgreSQL is now the selected storage provider. Put the connection string in
`.env`:

```powershell
$env:LAW_API_STORAGE_PROVIDER = "postgresql"
$env:LAW_API_POSTGRES_DSN = "postgresql://postgres:postgres@127.0.0.1:5432/legal"
```

Production storage providers are selected with `LAW_API_STORAGE_PROVIDER`:

```powershell
$env:LAW_API_STORAGE_PROVIDER = "postgresql"
# or: $env:LAW_API_STORAGE_PROVIDER = "qdrant"
```

Embedding aliases are:

- `gate-arabert-v1`
- `arabert-all-nli-triplet-matryoshka`
- `bge-m3`

Run the same PDF once for each embedding alias and storage provider. Each embed
request creates an MLflow run in `legal-rag-embeddings`, with parameters for embedding
model, storage provider, article count, and chunk count, plus metrics for
embedding dimension and embedded chunk count.

```powershell
$env:LAW_API_EMBEDDING = "gate-arabert-v1"
$env:LAW_API_STORAGE_PROVIDER = "qdrant"
uv run uvicorn law_api.main:app --host 127.0.0.1 --port 8000
```

Repeat with the other two aliases. Set `MLFLOW_TRACKING_URI` to the MLflow
server URL when the tracking server is not at its default.

## Run MLflow locally with UV

```powershell
uv sync --extra mlops
$env:MLFLOW_BACKEND_STORE = "file:./storage/mlruns"
$env:MLFLOW_ARTIFACT_ROOT = "./storage/mlartifacts"
uv run mlflow server --host 127.0.0.1 --port 5001 `
  --backend-store-uri $env:MLFLOW_BACKEND_STORE `
  --default-artifact-root $env:MLFLOW_ARTIFACT_ROOT
```

In another terminal:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
uv run uvicorn law_api.main:app --reload --host 127.0.0.1 --port 8000
```

On Windows, do not pass `--workers`; MLflow uses Waitress.
Open the MLflow UI at `http://127.0.0.1:5001`.

## Compare embedding models

The comparison script creates one MLflow run per model. It does not use the API,
PostgreSQL, or Qdrant; it only embeds the prepared chunks and records metrics.

```powershell
uv run python scripts/compare_embeddings.py
```

Available aliases:

```text
gate-arabert-v1
arabert-all-nli-triplet-matryoshka
bge-m3
```

Useful options:

```powershell
uv run python scripts/compare_embeddings.py `
  --models bge-m3 gate-arabert-v1 `
  --limit 100 `
  --batch-size 16 `
  --tracking-uri http://127.0.0.1:5001 `
  --experiment embedding-model-comparison
```

Each run logs the model ID, chunk count, batch size, vector dimension, total
embedding time, and chunks per second. The MLflow UI can compare these runs in
the `embedding-model-comparison` experiment.

---

## Multi-Tenant SaaS Endpoints

### 1. Authentication & Organization Onboarding

```bash
# Register a new workspace + owner account
curl -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"org_name":"Cairo Legal","email":"owner@example.com","password":"StrongPassword123","plan":"pro"}'

# Authenticate existing user
curl -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"owner@example.com","password":"StrongPassword123"}'

# Inspect active tenant profile (pass Authorization: Bearer <jwt> or X-API-Key: <key>)
curl http://127.0.0.1:8000/api/v1/auth/me -H "Authorization: Bearer <access_token>"
```

### 2. Scoped API Keys, Usage Metering & Audit Logs

```bash
# Create a programmatic API key (admin/owner role)
curl -X POST http://127.0.0.1:8000/api/v1/tenants/api-keys \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"name":"backend-worker","scopes":["read","write"]}'

# Inspect token usage, cost, and quota utilization
curl http://127.0.0.1:8000/api/v1/tenants/usage -H "X-API-Key: lrag_live_..."

# Query audit logs (admin/owner role)
curl http://127.0.0.1:8000/api/v1/tenants/audit-logs -H "Authorization: Bearer <access_token>"
```

### 3. Asynchronous Document Ingestion (`202 Accepted`) & Lifecycle

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/ingest \
  -H "Authorization: Bearer <access_token>" \
  -H "Idempotency-Key: civil-code-v1" \
  -F "file=@src/data/raw/egyptian_civil_code.pdf"

curl http://127.0.0.1:8000/api/v1/documents/jobs/<job_id> -H "Authorization: Bearer <access_token>"
curl http://127.0.0.1:8000/api/v1/documents -H "Authorization: Bearer <access_token>"
curl -X DELETE http://127.0.0.1:8000/api/v1/documents/1 -H "Authorization: Bearer <access_token>"
```

### 4. Multi-Turn Conversations & Feedback

```bash
# Create conversation thread
curl -X POST http://127.0.0.1:8000/api/v1/conversations \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"title":"Contract Nullity Review"}'

# Ask a contextualized follow-up turn
curl -X POST http://127.0.0.1:8000/api/v1/conversations/<conversation_id>/messages \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does Article 147 say?","top_k":5}'

# Submit answer feedback (score: 0 or 1)
curl -X POST http://127.0.0.1:8000/api/v1/feedback \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"trace_id":"<trace_id>","score":1,"comment":"Accurate article citation"}'
```