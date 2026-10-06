"""PII guardrail for /ask responses.

Presidio was already a declared dependency but nothing ever called it, so no
response was ever scanned. Worse, presidio needs a spaCy model, and spaCy models
are published only on GitHub releases — unreachable behind the TLS-intercepting
proxy on this machine. A presidio-only guardrail would therefore have looked
installed while silently never running.

So there are two engines:

* ``SpacyEngine`` — presidio, used when a spaCy model is actually present.
* ``RegexEngine`` — always available, no model, no network. It is what actually
  protects the endpoint in the deployed image.

Which entities matter here: answers are generated from a fixed statute corpus, so
the realistic PII risk is contact details leaking through a retrieved chunk, not
adversarial NER. Patterns cover exactly that, and are precise enough to keep
article numbers intact — the failure mode of over-broad PII masking is an
unreadable legal answer.

Streaming needs care: a token stream cannot be un-sent, so ``StreamRedactor``
withholds text until a sentence boundary has been scanned and cleared.
"""

import os
import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

REDACTION = "[REDACTED]"

# Deliberately not the full presidio entity list. Article numbers, dates and
# legal terms must survive redaction.
SUPPORTED_ENTITIES = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    # A run of digits and phone separators, then validated by digit count below.
    # Matching the whole run first and validating afterwards avoids stopping
    # mid-number, which used to leave trailing digits unredacted. The trailing
    # guard is `(?!\d)` and not a word boundary: a sentence-final number is
    # followed by ".", and a stricter guard backtracks the match instead.
    ("PHONE_NUMBER", re.compile(r"(?<![\w.])\+?\d[\d\s().-]{5,22}\d(?!\d)")),
    # 13-19 digits with optional separators; Luhn-checked before it counts.
    ("CREDIT_CARD", re.compile(r"(?<![\d-])(?:\d[ -]?){13,19}(?![\d-])")),
    ("IBAN_CODE", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("URL", re.compile(r"\bhttps?://[^\s<>\"')]+")),
)

_LUHN = "01234567890"


def _luhn_ok(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


@dataclass
class RedactionResult:
    text: str
    entities: dict[str, int] = field(default_factory=dict)
    scanned: bool = True

    @property
    def redacted(self) -> bool:
        return bool(self.entities)


def _looks_like_phone(candidate: str) -> bool:
    """Reject version numbers, dates, article counts and ranges as phones.

    A legal answer is dense with numbers ("Articles 147 and 148", "2-4
    sentences", "2019"), so a permissive digit-run pattern would redact the
    citations that make the answer useful.
    """
    digits = re.sub(r"\D", "", candidate)
    if not 7 <= len(digits) <= 15:
        return False
    # Pure years, and short ranges, are prose rather than contact details.
    if re.fullmatch(r"(19|20)\d{2}", digits):
        return False
    if "-" in candidate and len(digits) <= 8:
        return False
    return True


# spaCy models are published only on GitHub releases, never on PyPI. Presidio's
# default NLP engine calls `spacy download` when the model is missing, which
# means constructing AnalyzerEngine on the request path can attempt a network
# download and fail. That failure killed /ask (500 + a client-side disconnect)
# on a machine whose TLS proxy blocks GitHub. The models are therefore probed
# for BEFORE presidio is constructed, so nothing ever reaches the network.
_SPACY_MODELS = ("en_core_web_sm", "en_core_web_lg", "en_core_web_md")


def _spacy_model_available() -> str | None:
    """Return the importable spaCy model name, or None. Never touches the network."""
    import importlib.util

    for name in _SPACY_MODELS:
        if importlib.util.find_spec(name) is not None:
            return name
    return None


class RegexEngine:
    """Pattern-based detector. No model, no download, always available."""

    name = "regex"

    def detect(self, text: str) -> list[tuple[int, int, str]]:
        if not text.strip():
            return []
        found: list[tuple[int, int, str]] = []
        for entity, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(0)
                # Digit-run patterns can swallow the separator that follows them,
                # which glued "[REDACTED]" to the next word.
                trimmed = value.rstrip(" -.")
                if not trimmed:
                    continue
                start, end = match.start(), match.start() + len(trimmed)
                if entity == "CREDIT_CARD":
                    digits = re.sub(r"\D", "", trimmed)
                    if not 13 <= len(digits) <= 19 or not _luhn_ok(digits):
                        continue
                if entity == "PHONE_NUMBER" and not _looks_like_phone(trimmed):
                    continue
                found.append((start, end, entity))
        # Drop overlaps so a value is not redacted twice.
        found.sort()
        deduped: list[tuple[int, int, str]] = []
        last_end = -1
        for start, end, entity in found:
            if start >= last_end:
                deduped.append((start, end, entity))
                last_end = end
        return deduped


class SpacyEngine:
    """Presidio-backed detector, used only when a spaCy model is installed."""

    name = "presidio"

    def __init__(self, entities: Iterable[str] = SUPPORTED_ENTITIES, model: str | None = None) -> None:
        from presidio_analyzer import AnalyzerEngine

        self._model = model
        self._analyzer = AnalyzerEngine()
        self._entities = [entity for entity in entities]

    def detect(self, text: str) -> list[tuple[int, int, str]]:
        if not text.strip():
            return []
        results = self._analyzer.analyze(text=text, entities=self._entities, language="en")
        return [(item.start, item.end, item.entity_type) for item in results]


class PIIDetector:
    """Guardrail facade: prefers presidio, falls back to the regex engine."""

    def __init__(self, entities: tuple[str, ...] = SUPPORTED_ENTITIES) -> None:
        self.entities = tuple(entities)
        self._engine = None
        self._engine_name = ""
        self._warned = False

    @property
    def engine_name(self) -> str:
        self._ensure_engine()
        return self._engine_name

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        # Probe before constructing: presidio would otherwise shell out to
        # `spacy download` mid-request, which fails behind a TLS-inspecting proxy
        # and takes the whole /ask down with it.
        model = _spacy_model_available()
        if model:
            try:
                self._engine = SpacyEngine(self.entities, model)
                self._engine_name = "presidio"
                return self._engine
            except Exception:
                if not self._warned:
                    self._warned = True
                    print(f"pii: presidio failed to load '{model}', using the regex engine")
        else:
            if not self._warned:
                self._warned = True
                print(
                    "pii: no spaCy model installed, using the built-in regex engine. "
                    "Install one for richer detection "
                    "(`python -m spacy download en_core_web_sm` at image build time)."
                )
        self._engine = RegexEngine()
        self._engine_name = "regex"
        return self._engine

    def detect(self, text: str) -> list[tuple[int, int, str]]:
        return self._ensure_engine().detect(text)

    def redact(self, text: str) -> RedactionResult:
        spans = self.detect(text)
        if not spans:
            return RedactionResult(text=text, scanned=bool(text.strip()))
        counts: dict[str, int] = {}
        for start, end, entity in sorted(spans, key=lambda item: item[0], reverse=True):
            text = text[:start] + REDACTION + text[end:]
            counts[entity] = counts.get(entity, 0) + 1
        return RedactionResult(text=text, entities=counts)


# End-of-sentence markers: a terminator followed by whitespace. The whitespace
# is required — an earlier version also accepted end-of-BUFFER, so the chunk
# "Contact ali." looked like a finished sentence and was released with the email
# still unverified. A trailing sentence with no whitespace is flushed after the
# loop instead.
_SENTENCE_END = re.compile(r"[.!?؟।]\s")

# A sentence-only flush leaves a one-sentence answer as a single chunk, which is
# indistinguishable from not streaming at all. So a long buffer is also released
# early — but only at a whitespace that cannot fall inside an entity. A partial
# email/phone/IBN always contains '@' or a digit run, so a tail carrying either
# is held back until the boundary proves it is complete.
_MAX_BUFFER = 220
_PARTIAL_ENTITY = re.compile(r"[@]|\d{4,}")


def _safe_cut(pending: str) -> int:
    """Longest prefix of `pending` that is verified AND cannot split an entity."""
    cut = 0
    for match in _SENTENCE_END.finditer(pending):
        cut = match.end()
    if cut:
        return cut
    if len(pending) < _MAX_BUFFER:
        return 0
    # No sentence boundary yet. Cut at the last whitespace, but only when the
    # withheld tail shows no sign of an unfinished entity.
    space = pending.rfind(" ")
    if space < 0:
        return 0
    if _PARTIAL_ENTITY.search(pending[space:]):
        return 0
    return space + 1


class StreamRedactor:
    """Async iterator that withholds tokens until they have been scanned.

    A token stream cannot be un-sent, so nothing is emitted until a sentence
    boundary has been scanned and cleared; the safe prefix is then released and
    the remainder stays buffered. If the model is cut off before any terminator
    arrives, everything flushes at the end — slower, but never unredacted.

    Accumulated entity counts live in `.entities` so the caller can record them.
    """

    def __init__(self, detector: PIIDetector) -> None:
        self.detector = detector
        self.entities: dict[str, int] = {}

    def guarded(self, source):
        """Wrap an async token iterator, yielding only scanned-and-clear text.

        A plain method, not `__aiter__`: the source iterator does not exist when
        the redactor is constructed, so the protocol hook cannot do the job.
        """
        return self._iterate(source)

    async def _iterate(self, source) -> AsyncIterator[str]:
        pending = ""
        async for chunk in source:
            pending += chunk
            cut = _safe_cut(pending)
            if not cut:
                continue
            safe, pending = pending[:cut], pending[cut:]
            yield self._release(safe)
        if pending:
            yield self._release(pending)

    def _release(self, text: str) -> str:
        result = self.detector.redact(text)
        for entity, count in result.entities.items():
            self.entities[entity] = self.entities.get(entity, 0) + count
        return result.text


_detector: PIIDetector | None = None


def get_detector() -> PIIDetector:
    """Process-wide detector; engine construction is not free."""
    global _detector
    if _detector is None:
        enabled = os.getenv("LAW_API_PII_GUARDRAIL", "true").lower() not in (
            "false",
            "0",
            "off",
            "no",
        )
        _detector = PIIDetector()
        if not enabled:
            print("pii: guardrail disabled via LAW_API_PII_GUARDRAIL")
    return _detector
