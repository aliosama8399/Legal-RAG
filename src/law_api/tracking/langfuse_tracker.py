from contextlib import contextmanager


class LangfuseTracker:
    """Langfuse observability for the RAG pipeline (best-effort).

    Traces retrieval + generation. An unreachable server or missing keys
    must never fail a request — every call is guarded.
    """

    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        self.host = host
        self.public_key = public_key
        self.secret_key = secret_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not (self.public_key and self.secret_key):
                return None
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=self.public_key, secret_key=self.secret_key, host=self.host
                )
            except Exception:
                return None
        return self._client

    @contextmanager
    def trace(self, name: str, **input_data):
        """Context manager yielding a Langfuse observation (or None when off)."""
        client = self._ensure_client()
        if client is None:
            yield None
            return
        try:
            with client.start_as_current_observation(name=name, input=input_data) as observation:
                yield observation
        except Exception:
            yield None

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
