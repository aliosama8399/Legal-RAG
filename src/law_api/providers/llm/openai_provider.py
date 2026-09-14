class OpenAIProvider:
    """OpenAI-compatible chat client; the same class targets self-hosted
    OpenAI-compatible servers such as vLLM by pointing base_url at them."""

    name = "openai"

    def __init__(self, model_id: str, api_key: str = "", base_url: str = "") -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key or "not-needed", base_url=self._base_url or None)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        client = self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
