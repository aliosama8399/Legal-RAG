"""Latency benchmark for /ask and /ask/stream.

The requirement is "latency before vs after quantization documented in
README.md". This produces the measured numbers so the README quotes real data
from this machine rather than a guess.

    # FP16 baseline
    python -m scripts.latency_bench --label fp16

    # after switching .env to the 4-bit configuration
    python -m scripts.latency_bench --label nf4-4bit

Results append to data/eval/latency_report.md, and the comparison table is
printed for pasting into the README.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

REPORT_PATH = PROJECT_ROOT / "data" / "eval" / "latency_report.md"
FALLBACK_QUESTIONS = [
    "What are the general provisions of the Egyptian Civil Code?",
    "Who is liable for damages caused by a thing?",
    "What is a contract of sale under Egyptian law?",
    "State the grounds for contract nullity.",
    "What are the rights of the creditor in Egyptian civil law?",
]


def load_questions(limit: int) -> list[str]:
    dataset = PROJECT_ROOT / "data" / "eval" / "qa_dataset.jsonl"
    if dataset.is_file():
        questions = [
            json.loads(line).get("question")
            for line in dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        questions = [q for q in questions if q]
        if questions:
            return questions[:limit]
    return FALLBACK_QUESTIONS[:limit]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


async def timed_post(url: str, payload: dict, timeout: float) -> tuple[float, int, str]:
    import httpx

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        elapsed = time.perf_counter() - started
    return elapsed, response.status_code, response.text[:200]


async def bench_sequential(url: str, questions: list[str], top_k: int, timeout: float) -> dict:
    latencies: list[float] = []
    statuses: list[int] = []
    errors = 0
    for question in questions:
        elapsed, status, _ = await timed_post(
            url, {"question": question, "top_k": top_k}, timeout
        )
        latencies.append(elapsed)
        statuses.append(status)
        if status != 200:
            errors += 1
    return _summarize(latencies, statuses, errors, concurrency=1)


async def bench_concurrent(url: str, questions: list[str], top_k: int, concurrency: int, timeout: float) -> dict:
    gate = asyncio.Semaphore(concurrency)

    async def one(index: int, question: str):
        async with gate:
            return await timed_post(url, {"question": question, "top_k": top_k}, timeout)

    started = time.perf_counter()
    results = await asyncio.gather(
        *(one(index, question) for index, question in enumerate(questions))
    )
    wall = time.perf_counter() - started
    latencies = [item[0] for item in results]
    statuses = [item[1] for item in results]
    errors = sum(1 for status in statuses if status != 200)
    summary = _summarize(latencies, statuses, errors, concurrency)
    summary["wall_seconds"] = round(wall, 3)
    summary["throughput_rps"] = round(len(questions) / wall, 3) if wall else 0.0
    return summary


def _summarize(latencies: list[float], statuses: list[int], errors: int, concurrency: int) -> dict:
    return {
        "requests": len(latencies),
        "concurrency": concurrency,
        "errors": errors,
        "mean_seconds": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "p50_seconds": round(percentile(latencies, 0.50), 3),
        "p95_seconds": round(percentile(latencies, 0.95), 3),
        "p99_seconds": round(percentile(latencies, 0.99), 3),
        "min_seconds": round(min(latencies), 3) if latencies else 0.0,
        "max_seconds": round(max(latencies), 3) if latencies else 0.0,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", default="run", help="Name for this configuration, e.g. fp16 or nf4-4bit")
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    url = f"{args.base_url.rstrip('/')}/api/v1/ask"
    print(f"benchmarking {args.label}: {len(questions)} questions against {url}")

    print("  sequential pass ...")
    sequential = await bench_sequential(url, questions, args.top_k, args.timeout)
    print(f"    p50={sequential['p50_seconds']}s p95={sequential['p95_seconds']}s")

    print(f"  concurrent pass ({args.concurrency} at a time) ...")
    concurrent = await bench_concurrent(url, questions, args.top_k, args.concurrency, args.timeout)
    print(f"    p95={concurrent['p95_seconds']}s throughput={concurrent.get('throughput_rps')} rps")

    record = {
        "label": args.label,
        "sequential": sequential,
        "concurrent": concurrent,
        "questions": len(questions),
        "top_k": args.top_k,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {args.label} ({len(questions)} questions, top_k={args.top_k})\n\n")
        handle.write("| pass | mean | p50 | p95 | p99 | errors |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for name, data in (("sequential", sequential), (f"concurrent x{args.concurrency}", concurrent)):
            handle.write(
                f"| {name} | {data['mean_seconds']}s | {data['p50_seconds']}s | "
                f"{data['p95_seconds']}s | {data['p99_seconds']}s | {data['errors']} |\n"
            )
    print(f"appended to {REPORT_PATH}")
    print(json.dumps(record, indent=2))

    total_errors = sequential["errors"] + concurrent["errors"]
    if total_errors:
        print(f"WARNING: {total_errors} non-200 responses — check the API is healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
