"""Locust stress test for the Egyptian Civil Code RAG API.

Targets the dockerized law-api (default http://localhost:8000).
Questions are sampled from the evaluation dataset when present,
otherwise from a small built-in set.

Run:
    uv run locust -f locustfile.py --host http://localhost:8000
    # headless stress profile (ramp 1 -> 50 users over 3 minutes):
    uv run locust -f locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 0.3 -t 3m --html locust_report.html

User classes and weights:
    SearchUser  (3) — POST /api/v1/search          (cheap, retrieval only)
    AskUser     (2) — POST /api/v1/ask             (full RAG: retrieval + generation)
    StreamUser  (1) — POST /api/v1/ask/stream      (SSE streaming RAG)
"""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, events, task

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "qa_dataset.jsonl"

FALLBACK_QUESTIONS = [
    "What is a contract according to the Egyptian Civil Code?",
    "What does article 147 say about the vendor's privilege?",
    "How are house rents secured by a privilege?",
    "What are the provisions for the sale of an immovable?",
    "What is the legal position of a co-owner after partition?",
]

QUESTIONS: list[str] = []
QUESTION_WEIGHTS: list[float] = []


@events.init.add_listener
def _load_questions(environment, **kwargs):
    global QUESTIONS, QUESTION_WEIGHTS
    dataset = []
    if DATASET_PATH.is_file():
        dataset = [
            json.loads(line)
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if dataset:
        QUESTIONS = [item["question"] for item in dataset]
        # Shorter questions are cheaper to embed/answer — weight by inverse length.
        QUESTION_WEIGHTS = [1.0 / max(len(q), 20) for q in QUESTIONS]
        print(f"locust: loaded {len(QUESTIONS)} questions from {DATASET_PATH}")
    else:
        QUESTIONS = FALLBACK_QUESTIONS
        QUESTION_WEIGHTS = [1.0] * len(QUESTIONS)
        print("locust: using built-in fallback questions")


def _pick_question() -> str:
    return random.choices(QUESTIONS, weights=QUESTION_WEIGHTS, k=1)[0]


class SearchUser(HttpUser):
    """Retrieval-only users — the cheapest endpoint; tolerates high concurrency."""

    weight = 3
    wait_time = between(0.5, 2.0)

    @task
    def search(self):
        self.client.post(
            "/api/v1/search",
            json={"question": _pick_question(), "top_k": 5},
            name="/api/v1/search",
        )


class AskUser(HttpUser):
    """Full RAG users — retrieval + generation; the generation model is the bottleneck."""

    weight = 2
    wait_time = between(2.0, 6.0)

    @task
    def ask(self):
        self.client.post(
            "/api/v1/ask",
            json={"question": _pick_question(), "top_k": 5},
            name="/api/v1/ask",
            timeout=120,
        )


class StreamUser(HttpUser):
    """Streaming RAG users — parses the SSE token stream end-to-end."""

    weight = 1
    wait_time = between(2.0, 6.0)

    @task
    def ask_stream(self):
        tokens = 0
        with self.client.post(
            "/api/v1/ask/stream",
            json={"question": _pick_question(), "top_k": 5},
            name="/api/v1/ask/stream",
            timeout=120,
            stream=True,
            catch_response=True,
        ) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[len("data: ") :])
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "token":
                    tokens += 1
                elif event.get("type") == "error":
                    response.failure(f"stream error: {event.get('detail')}")
                    return
            if tokens == 0:
                response.failure("stream completed without tokens")
            else:
                response.request_meta["response_length"] = tokens
