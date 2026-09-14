# Data API

The API is organized as MVC-style layers:

- `law_api/controllers/data/upload_data_controller.py`: upload and embedding controller.
- `law_api/routes/data_routes.py`: data routes.
- `law_api/services.py`: separate chunking and embedding use cases.
- `law_api/providers/storage/`: PostgreSQL and Qdrant providers.
- `law_api/schemas.py`: request/response contracts.
- `law_api/main.py`: application composition.

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