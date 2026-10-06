"""Regression check for the streaming PII guardrail.

Run it after touching StreamRedactor:

    python -m scripts.check_pii_stream

It asserts the two properties that actually matter, both of which have been
broken at least once:

1. An entity is never split across frames. A redactor that releases a partial
   email has leaked it, even if the total text looks redacted afterwards.
2. A long answer releases more than once. Sentence-only flushing made a
   one-sentence answer indistinguishable from not streaming at all.
"""

import asyncio
import sys

sys.path.insert(0, "src")
from law_api.guardrails.pii import PIIDetector, StreamRedactor


async def run(text, label):
    async def src():
        for i in range(0, len(text), 12):
            yield text[i : i + 12]

    redactor = StreamRedactor(PIIDetector())
    frames = [chunk async for chunk in redactor.guarded(src())]
    print(f"{label}: {len(frames)} frame(s)")
    for frame in frames:
        print("   ", repr(frame))
    return frames


async def main() -> int:
    failures = []

    frames = await run(
        "Contact ali.osama@example.com or call +20 100 123 4567 today.",
        "PII - must never split an entity",
    )
    joined = "".join(frames)
    if "ali.osama" in joined or "example.com" in joined:
        failures.append("an entity leaked through the stream")
    if any("REDACTED" not in frame for frame in frames):
        failures.append("a frame carrying PII was not redacted")

    frames = await run(
        "The vendor is liable for damages caused by a defective thing sold under "
        "warranty per Article 450 of the Civil Code. The buyer may sue. Damages "
        "are due.",
        "multi-sentence",
    )
    if len(frames) < 2:
        failures.append("a multi-sentence answer was not streamed progressively")

    frames = await run(
        "Liability attaches to the custodian of a thing under Article 1000 of the "
        "Egyptian Civil Code, subject to proof of a defect at the time of custody. "
        "The custodian may rebut with diligence. Courts apply the ordinary burden. "
        "This makes the answer longer than the buffer threshold used by the "
        "redactor, so it must release more than once.",
        "long answer crossing the buffer threshold",
    )
    if len(frames) < 2:
        failures.append("a long answer was withheld until the end")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print("  -", failure)
        return 1
    print("\nOK: entities intact, answers stream progressively")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
