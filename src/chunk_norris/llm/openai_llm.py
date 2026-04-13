import os

import openai  # openai==2.29.0

from chunk_norris.llm.base import BaseLLM, LLMError

# Reasoning models do not accept temperature or top_p.
# They use reasoning_effort instead.
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "o5")


def _is_reasoning_model(model: str) -> bool:
    """Returns True if the model is a reasoning model that does not support temperature."""
    name = model.lower().strip()
    return any(name.startswith(prefix) for prefix in _REASONING_MODEL_PREFIXES)


class OpenAILLM(BaseLLM):
    """
    OpenAI implementation of BaseLLM.

    Supports both standard chat models (gpt-4o, gpt-4o-mini, etc.) and
    reasoning models (o1, o3, o4-mini, etc.). The difference is handled
    automatically based on the model name — reasoning models do not accept
    temperature and use reasoning_effort instead.

    The API key is read from the OPENAI_API_KEY environment variable by
    default — never hardcode API keys in your code.

    Args:
        model (str): The OpenAI model to use. Default: "gpt-4o".
                     Examples:
                         Standard:  "gpt-4o", "gpt-4o-mini"
                         Reasoning: "o4-mini-2025-04-16", "o3", "o1"
        reasoning_effort (str): Only used for reasoning models. Controls the
                                 depth of internal reasoning. Options: "low",
                                 "medium", "high". Default: "medium".
        api_key (str | None): OpenAI API key. If None, reads from the
                              OPENAI_API_KEY environment variable.

    Raises:
        ValueError: If no API key is found in arguments or environment.

    Example::

        # Standard model — temperature is used
        llm = OpenAILLM(model="gpt-4o")
        response = llm.generate("What is RAG?", temperature=0.3)

        # Reasoning model — temperature is ignored, reasoning_effort is used
        llm = OpenAILLM(model="o4-mini-2025-04-16", reasoning_effort="medium")
        response = llm.generate("What is RAG?")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        reasoning_effort: str = "medium",
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key found. Either pass api_key= or set the "
                "OPENAI_API_KEY environment variable."
            )
        if reasoning_effort not in ("low", "medium", "high"):
            raise ValueError(
                f"reasoning_effort must be 'low', 'medium', or 'high', "
                f"got '{reasoning_effort}'"
            )

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.is_reasoning_model = _is_reasoning_model(model)
        self._client = openai.OpenAI(api_key=resolved_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Sends a prompt to the OpenAI Responses API and returns the response text.

        For standard models, temperature and max_tokens are passed as-is.
        For reasoning models, temperature is silently ignored and reasoning_effort
        is used instead — no error is raised, the call just works.

        Args:
            prompt (str): The full prompt to send to the model.
            temperature (float): Sampling temperature. Ignored for reasoning models.
                                 Default: 0.0.
            max_tokens (int): Maximum tokens in the response. Default: 1024.

        Returns:
            str: The model's response text.

        Raises:
            LLMError: If the API call fails for any reason.
        """
        try:
            if self.is_reasoning_model:
                response = self._client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": self.reasoning_effort},
                )
            else:
                response = self._client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            return response.output_text

        except openai.APIConnectionError as e:
            raise LLMError(
                message="Could not connect to the OpenAI API. Check your internet connection.",
                provider="OpenAI",
                original_error=e,
            ) from e

        except openai.AuthenticationError as e:
            raise LLMError(
                message="Invalid OpenAI API key.",
                provider="OpenAI",
                original_error=e,
            ) from e

        except openai.RateLimitError as e:
            raise LLMError(
                message="OpenAI rate limit reached. Try again later or reduce request frequency.",
                provider="OpenAI",
                original_error=e,
            ) from e

        except openai.APIStatusError as e:
            raise LLMError(
                message=f"OpenAI API returned status {e.status_code}.",
                provider="OpenAI",
                original_error=e,
            ) from e
