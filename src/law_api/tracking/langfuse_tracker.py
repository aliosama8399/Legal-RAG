import os
from contextlib import contextmanager

# The Langfuse SDK's start_as_current_observation() takes a fixed keyword list
# and NO **kwargs. Passing arbitrary attributes to it raises TypeError, which is
# why tracing used to fail silently and /ask produced no trace at all. Callers
# may therefore pass whatever is convenient; these rules route it onto the real
# parameters instead of dropping it.
#
#   input    -> the content the step worked on (question text, prompt)
#   model    -> the LLM that produced the output
#   model_parameters -> generation settings
#   metadata -> everything descriptive that is not content
_INPUT_KEYS = frozenset({"question", "prompt", "query", "text", "input", "answer"})
_MODEL_KEYS = frozenset({"model", "llm_model", "model_id"})
_MODEL_PARAM_KEYS = frozenset({"temperature", "max_tokens", "top_p", "stream"})


def _split_attributes(attributes: dict) -> tuple[dict, dict, dict | None, dict]:
    """Route caller attributes onto input / metadata / model / model_parameters."""
    payload: dict = {}
    metadata: dict = {}
    model_parameters: dict = {}
    model: str | None = None
    for key, value in attributes.items():
        if value is None:
            continue
        if key in _INPUT_KEYS:
            payload[key] = value
        elif key in _MODEL_KEYS:
            model = str(value)
        elif key in _MODEL_PARAM_KEYS:
            model_parameters[key] = value
        else:
            metadata[key] = value
    return payload, metadata, model, model_parameters


class LangfuseTracker:
    """Langfuse observability for the RAG pipeline (best-effort).

    Uses the OpenTelemetry-based Python SDK v4 surface: ``base_url`` plus
    ``start_as_current_observation(as_type=...)``. Every call is guarded — an
    unreachable server or missing keys must never fail a request.
    """

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        expected_project: str = "",
    ) -> None:
        self.host = host
        self.public_key = public_key
        self.secret_key = secret_key
        self.expected_project = expected_project
        self._client = None
        self._warned_keys: set[str] = set()

    def _ensure_client(self):
        if self._client is None:
            if not (self.public_key and self.secret_key):
                return None
            try:
                from langfuse import Langfuse

                # v4 spells the endpoint `base_url`; `host` is the legacy alias.
                # Tracing must be initialised AFTER the env is loaded, otherwise
                # the client latches onto the wrong credentials/host.
                client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    base_url=self.host,
                )
                # Wrong keys/host otherwise fail silently and the UI just stays
                # empty. A wrong PROJECT is just as silent, so check for it too.
                client.auth_check()
                if self.expected_project:
                    names = [project.name for project in client.api.projects.get().data]
                    if self.expected_project not in names:
                        print(
                            f"langfuse keys belong to project(s) {names}, "
                            f"not '{self.expected_project}' - not tracing"
                        )
                        return None
            except Exception as error:
                print(f"langfuse unavailable at {self.host} (check host and keys): {error}")
                return None
            self._client = client
        return self._client

    def _observation_kwargs(self, name: str, as_type: str, attributes: dict) -> dict:
        payload, metadata, model, model_parameters = _split_attributes(attributes)
        kwargs: dict = {"name": name, "as_type": as_type}
        if payload:
            # The SDK takes exactly one `input`; keep the caller's key layout.
            kwargs["input"] = payload
        if metadata:
            kwargs["metadata"] = metadata
        if model:
            kwargs["model"] = model
        if model_parameters:
            kwargs["model_parameters"] = model_parameters
        return kwargs

    @contextmanager
    def trace(self, name: str, as_type: str = "span", **attributes):
        """Live span around real work; yields a Langfuse observation, or None when off.

        The span brackets the work itself, so the trace is visible while the
        request is still running. Tracing errors are swallowed, but an error
        raised by the wrapped work is always re-raised and still marked on the
        span. Child observations opened inside this block nest automatically
        through the active OpenTelemetry context.
        """
        client = self._ensure_client()
        if client is None:
            yield None
            return

        # BentoML's ASGI server sets an unsampled parent span (sampled=0) on incoming
        # requests. OpenTelemetry would drop child spans created under an unsampled parent.
        # If the active span is non-recording, detach it so Langfuse starts a recording root trace.
        clean_token = None
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace as otel_trace

            current_span = otel_trace.get_current_span()
            if (
                current_span
                and not current_span.is_recording()
                and current_span != otel_trace.INVALID_SPAN
            ):
                clean_token = otel_context.attach(
                    otel_trace.set_span_in_context(otel_trace.INVALID_SPAN)
                )
        except Exception:
            clean_token = None

        try:
            kwargs = self._observation_kwargs(name, as_type, attributes)
            span = client.start_as_current_observation(**kwargs)
            observation = span.__enter__()
        except Exception as error:
            # Loud on the first failure, then quiet: this used to be swallowed
            # completely, which is how a broken tracer looked like a healthy one.
            if "trace-open" not in self._warned_keys:
                self._warned_keys.add("trace-open")
                print(f"langfuse: could not open span '{name}' ({error}) - tracing degraded")
            if clean_token is not None:
                try:
                    otel_context.detach(clean_token)
                except Exception:
                    pass
            yield None
            return
        try:
            yield observation
        except BaseException as error:
            try:
                span.__exit__(type(error), error, error.__traceback__)
            except Exception:
                pass
            raise
        else:
            try:
                span.__exit__(None, None, None)
            except Exception:
                pass
        finally:
            if clean_token is not None:
                try:
                    otel_context.detach(clean_token)
                except Exception:
                    pass

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass
