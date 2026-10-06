"""BentoML wrapper around the RAG API.

BentoML serves the API as an ASGI application; the generative model runs
on vLLM (OpenAI-compatible) configured via LAW_API_LLM_PROVIDER=vllm.

Build:  bentoml build
Serve:  bentoml serve law-rag:latest --port 3000

The FastAPI app (routes under /api/v1/...) is mounted at "/", so all
endpoints (/api/v1/documents/upload, /api/v1/ask/stream, ...) work through
the BentoML HTTP server unchanged.

NOTE: @bentoml.asgi_app MUST decorate the CLASS (below @bentoml.service) -
the service factory reads __bentoml_mounted_apps__ from the class object;
decorating a method stores it on the function and the mount is never applied
(all routes 404).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import bentoml

from law_api.main import app as fastapi_app


@bentoml.service(
    name="law-rag",
    # ONE worker process = the embedding model + reranker are loaded ONCE.
    # Every extra worker would load its own full copy into RAM.
    workers=1,
    # RAG requests (retrieval + rerank + generation on a small GPU) exceed
    # BentoML's default 60s traffic timeout -> 504 Gateway Timeout.
    # concurrency 4 (was 8): each in-flight request runs a CPU cross-encoder
    # rerank, so 8 at once spikes CPU and RAM; vLLM batches generation anyway.
    traffic={"timeout": 600, "concurrency": 4},
)
@bentoml.asgi_app(fastapi_app, path="/")
class LawRAG:
    """Serves the Egyptian Civil Code RAG; generation is delegated to vLLM."""