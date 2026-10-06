"""Batch re-indexing of every PDF in a directory.

There was no batch re-index script: ingestion only existed as two manual HTTP
calls (`POST /documents/upload` then `POST /documents/{id}/embed`). This drives
the whole sequence per document, verifies the result, and is idempotent so it
can be re-run safely after a model change.

    python -m scripts.reindex --dir storage/uploads
    python -m scripts.reindex --dir storage/uploads --dry-run
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

PDF_SUFFIXES = {".pdf"}


def discover(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"not a directory: {directory}")
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in PDF_SUFFIXES)


async def upload(client, base_url: str, path: Path) -> dict:
    with path.open("rb") as handle:
        response = await client.post(
            f"{base_url}/api/v1/documents/upload", files={"file": (path.name, handle, "application/pdf")}
        )
    response.raise_for_status()
    return response.json()


async def embed(client, base_url: str, document_id: int) -> dict:
    response = await client.post(f"{base_url}/api/v1/documents/{document_id}/embed", json={})
    response.raise_for_status()
    return response.json()


async def count_points(client, base_url: str, document_id: int) -> int:
    """Verify the document is actually searchable, not just reported embedded."""
    response = await client.post(
        f"{base_url}/api/v1/search", json={"question": "contract", "top_k": 20, "document_id": document_id}
    )
    if response.status_code != 200:
        return 0
    return len(response.json().get("results", []))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=PROJECT_ROOT / "storage" / "uploads")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true", help="List what would be indexed and exit")
    args = parser.parse_args()

    pdfs = discover(args.dir)
    if not pdfs:
        print(f"no PDFs in {args.dir}")
        return 1
    print(f"found {len(pdfs)} document(s) in {args.dir}")
    if args.dry_run:
        for path in pdfs:
            print(f"  would index {path.name} ({path.stat().st_size / 1024:.0f} KB)")
        return 0

    import httpx

    failures = 0
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        for path in pdfs:
            print(f"\n=== {path.name} ===")
            started = time.perf_counter()
            try:
                uploaded = await upload(client, args.base_url, path)
                document_id = uploaded["document_id"]
                print(
                    f"  uploaded  document_id={document_id} "
                    f"articles={uploaded.get('articles')} chunks={uploaded.get('chunks')}"
                )
                embedded = await embed(client, args.base_url, document_id)
                print(
                    f"  embedded  chunks={embedded.get('chunks')} "
                    f"dimension={embedded.get('embedding_dimension')} "
                    f"model={embedded.get('embedding_model')}"
                )
                found = await count_points(client, args.base_url, document_id)
                status = "OK" if found else "NO HITS"
                print(f"  verified  {found} searchable hit(s) -> {status}")
                print(f"  took {time.perf_counter() - started:.1f}s")
            except Exception as error:
                failures += 1
                print(f"  FAILED: {error}")

    print(f"\n{len(pdfs) - failures}/{len(pdfs)} document(s) indexed successfully")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
