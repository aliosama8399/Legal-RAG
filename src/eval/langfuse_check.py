"""Langfuse-only smoke test: emits one trace with nested observations and
verifies it can be read back through the public API.

Does not touch the RAG pipeline, Qdrant or vLLM — it only proves that the
credentials, host and ingestion path work end to end.

    docker compose --profile eval run --rm eval-job python -m eval.langfuse_check
"""

import base64
import json
import time
import urllib.parse
import urllib.request
import uuid

from langfuse import Langfuse


def main() -> int:
    from law_api.config import settings

    print(f"host={settings.langfuse_host}")
    print(f"public_key={settings.langfuse_public_key[:14]}...")
    if not settings.langfuse_project_name:
        print("LANGFUSE_PROJECT_NAME is empty - the wrong-project guard is disabled")
    else:
        print(f"expected project={settings.langfuse_project_name!r}")

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_host,
    )

    try:
        client.auth_check()
        print("auth_check: OK")
    except Exception as error:
        print(f"auth_check FAILED: {error}")
        return 1

    projects = [project.name for project in client.api.projects.get().data]
    print(f"projects visible to these keys: {projects}")
    expected = settings.langfuse_project_name
    if expected and expected not in projects:
        print(f"WRONG PROJECT: keys do not belong to {expected!r}")
        return 1
    project_id = next(
        (project.id for project in client.api.projects.get().data if project.name == expected),
        "",
    )

    marker = f"langfuse-check-{uuid.uuid4().hex[:8]}"
    print(f"emitting trace {marker}")

    with client.start_as_current_observation(
        as_type="span", name="langfuse-check", input={"question": marker}
    ) as root:
        with client.start_as_current_observation(
            as_type="retriever", name="vector-search", input={"question": marker}
        ) as retriever:
            retriever.update(output={"sources": ["Article 1", "Article 147"]})

        with client.start_as_current_observation(
            as_type="generation", name="answer-generation", model="smoke-test"
        ) as generation:
            generation.update(output={"answer": "ok"})

        root.update(output={"answer": "ok"})
        root.score(name="smoke_test", value=1.0)
        trace_id = root.trace_id or client.get_current_trace_id()

    # Ingestion is asynchronous: flush() guarantees delivery, not read visibility.
    client.flush()
    print("flushed; polling the v4 read API (docs say 15-30s)...")

    # v4 reads through the Observations API v2. The legacy GET /api/public/traces
    # and api.trace.get() read the v3-compat `traces` table, which stays EMPTY on
    # a healthy v4 deployment — a 404 there does not mean ingestion failed.
    query = urllib.parse.urlencode({"traceId": trace_id, "limit": 10})
    url = f"{settings.langfuse_host.rstrip('/')}/api/public/v2/observations?{query}"
    auth = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()
    request = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})

    for attempt in range(1, 7):
        time.sleep(5)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.load(response)
        except Exception as error:  # noqa: BLE001 - just keep polling
            print(f"  attempt {attempt}: not readable yet ({error})")
            continue
        rows = payload.get("data") or []
        if rows:
            print(f"  attempt {attempt}: TRACE VISIBLE via Observations API v2")
            for row in rows:
                print(f"    - {row.get('type', '?'):<10} {row.get('name')}")
            print(f"  UI: {settings.langfuse_host.rstrip('/')}/project/{project_id}")
            return 0

    print(f"trace {trace_id} was flushed but never became readable - check the worker")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
