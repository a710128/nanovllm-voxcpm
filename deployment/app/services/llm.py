from __future__ import annotations

import threading
import time
from typing import Any, AsyncIterator

from app.core.config import LLMConfig
from app.core.metrics import PIPELINE_STAGE_DURATION_SECONDS

import openai
class LLMClient:
    """Async streaming client for OpenAI API."""

    def __init__(self, config: LLMConfig) -> None:
        self.api_key = config.api_key
        self.model = config.model
        self.timeout = config.timeout
        self.prompt_template = config.prompt_template
        self._client = None
        self._init_lock = threading.Lock()

    def warmup(self) -> None:
        """Initialize the underlying SDK/model object.

        Intended to be called at server startup so the first request doesn't pay
        import/configuration/model-construction overhead.
        """
        _ = self._get_client()

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is not None:
            return self._client

        with self._init_lock:
            if self._client is not None:
                return self._client
            
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
            return self._client

    def _build_prompt(self, user_input: str, prompt_template_override: str | None = None) -> str:
        """Build the final prompt with system template and user input.

        Args:
            user_input: The user's message (raw transcript).
            prompt_template_override: If set, use this template instead of the default.
        """
        template = prompt_template_override if prompt_template_override else self.prompt_template
        return f"{template}\nstudent: {user_input}\nteacher:"

    async def stream(
        self,
        user_input: str,
        prompt_template_override: str | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> AsyncIterator[str]:
        """Stream LLM response chunks for the given user input.

        Args:
            user_input: The user's message (raw transcript).
            prompt_template_override: If set, use this speaker-specific prompt template.
            interrupt_event: Signal to abort stream mid-generation.

        Yields:
            Text chunks as they are generated.

        Raises:
            RuntimeError: If the LLM API call fails.
        """
        final_prompt = self._build_prompt(user_input, prompt_template_override)
        t0 = time.perf_counter()
        first_chunk = True
        response = None

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                stream=True,
            )

            async for chunk in response:
                if interrupt_event is not None and interrupt_event.is_set():
                    break

                if first_chunk:
                    PIPELINE_STAGE_DURATION_SECONDS.labels(stage="llm_ttft").observe(
                        time.perf_counter() - t0
                    )
                    first_chunk = False

                content = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                if content:
                    yield content

        except Exception as e:
            raise RuntimeError(f"LLM API error: {e}") from e
        finally:
            PIPELINE_STAGE_DURATION_SECONDS.labels(stage="llm").observe(time.perf_counter() - t0)

    async def generate(self, user_input: str, prompt_template_override: str | None = None) -> str:
        """Generate a complete LLM response (non-streaming).

        Args:
            user_input: The user's message (raw transcript).
            prompt_template_override: If set, use this speaker-specific prompt template.

        Returns:
            Complete generated text.
        """
        chunks: list[str] = []
        async for chunk in self.stream(user_input, prompt_template_override):
            chunks.append(chunk)
        return "".join(chunks)
