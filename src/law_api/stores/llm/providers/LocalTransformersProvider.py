import asyncio

from ..LLMInterface import LLMInterface


class LocalTransformersProvider(LLMInterface):
    """Offline text generation using a local Hugging Face model.

    The transformers pipeline is blocking CPU/GPU work, so it runs in a
    worker thread to keep the FastAPI event loop responsive.
    """

    name = "local"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline("text-generation", model=self.model_id)
        return self._pipeline

    def _generate_sync(self, prompt, *, system, temperature, max_tokens) -> str:
        text_pipeline = self._ensure_pipeline()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        generate_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        output = text_pipeline(full_prompt, **generate_kwargs)
        generated = output[0]["generated_text"]
        return generated[len(full_prompt) :].strip() if generated.startswith(full_prompt) else generated.strip()

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync, prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )

    async def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        # The transformers pipeline has no native async streaming; yield the
        # whole answer as a single chunk so the /ask/stream contract holds.
        yield await self.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
