from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    All LLM integrations (OpenAI, Anthropic, Gemini, etc.) must inherit
    from this class and implement the generate() method. This keeps the
    evaluator completely decoupled from any specific LLM API.

    Temperature and max_tokens are defined at call time in generate(), not
    at construction time. This is intentional — different parts of chunk-norris
    use the same LLM object with different settings:
        - answer generation: higher temperature for more natural responses
        - scoring / evaluation: temperature=0.0 for deterministic results
        - question generation: higher temperature for more varied questions

    Example of implementing a custom provider::

        from chunk_norris.llm.base import BaseLLM, LLMError

        class MyLLM(BaseLLM):
            def generate(
                self,
                prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 1024,
            ) -> str:
                try:
                    response = my_api.call(prompt, temperature, max_tokens)
                    return response.text
                except my_api.Error as e:
                    raise LLMError(
                        message="My API call failed.",
                        provider="MyProvider",
                        original_error=e,
                    ) from e

    Then pass it to Norris::

        from chunk_norris import Norris
        norris = Norris(llm=MyLLM())
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """
        Sends a prompt to the LLM and returns the response as a string.

        Args:
            prompt (str): The full prompt to send to the model.
            temperature (float): Sampling temperature between 0.0 and 2.0.
                                 Lower = more deterministic. Default: 0.0.
            max_tokens (int): Maximum number of tokens in the response.
                              Default: 1024.

        Returns:
            str: The model's response text.

        Raises:
            LLMError: If the API call fails for any reason.
        """
        pass


class LLMError(Exception):
    """
    Raised when an LLM API call fails.

    Wraps provider-specific exceptions into a single exception type
    so the evaluator does not need to handle OpenAI, Anthropic, or
    Gemini errors separately.

    Attributes:
        message (str): Human-readable description of the failure.
        provider (str): Name of the LLM provider that raised the error.
        original_error (Exception): The original exception from the provider SDK.
    """

    def __init__(self, message: str, provider: str, original_error: Exception) -> None:
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error

    def __str__(self) -> str:
        return (
            f"[{self.provider}] {super().__str__()} "
            f"(original: {type(self.original_error).__name__}: {self.original_error})"
        )