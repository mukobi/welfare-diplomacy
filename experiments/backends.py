"""Language model backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
from typing import Optional

import backoff
import openai
from openai.error import RateLimitError


@dataclass
class ModelResponse:
    """A response from a model for a single turn of actions."""

    model_name: str  # Name of the generating model.
    reasoning: str  # Private reasoning to generate the response.
    orders: list[str]  # Orders to execute.
    messages: dict[str, str]  # Messages to send to other powers.
    system_prompt: str  # System prompt
    user_prompt: Optional[str]  # User prompt if available
    prompt_tokens: int  # Number of tokens in prompt
    completion_tokens: int  # Number of tokens in completion
    total_tokens: int  # Total number of tokens in prompt and completion
    completion_time_sec: float  # Time to generate completion in seconds


class LanguageModelBackend(ABC):
    """Abstract class for language model backends."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        """
        Complete a prompt with the language model backend.

        Returns the completion as a dict.
        """
        raise NotImplementedError("Subclass must implement abstract method")


class OpenAIChatBackend(LanguageModelBackend):
    """OpenAI chat completion backend (for GPT-4, GPT-3.5-turbo)."""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def complete(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> ModelResponse:
        try:
            response = self.completions_with_backoff(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            completion = response.choices[0].message.content  # type: ignore
            self.logger.debug("Completion:\n%s", completion)
            completion = json.loads(completion, strict=False)
            # Turn recipients in messages into ALLCAPS for the engine
            completion["messages"] = {
                recipient.upper(): message
                for recipient, message in completion["messages"].items()
            }
            assert "usage" in response, "OpenAI response does not contain usage"
            usage = response["usage"]  # type: ignore
            completion_time_sec = response.response_ms / 1000.0  # type: ignore
            return ModelResponse(
                model_name=self.model_name,
                reasoning=completion["reasoning"],
                orders=completion["orders"],
                messages=completion["messages"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                completion_time_sec=completion_time_sec,
            )

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(
                "Error completing prompt ending in\n%s\n\nException:\n%s",
                user_prompt[-300:],
                exc,
            )
            raise

    @backoff.on_exception(backoff.expo, RateLimitError)
    def completions_with_backoff(self, **kwargs):
        """Exponential backoff for OpenAI API rate limit errors."""
        response = openai.ChatCompletion.create(**kwargs)
        assert response is not None, "OpenAI response is None"
        assert "choices" in response, "OpenAI response does not contain choices"
        return response
