"""Language model backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging

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

    def complete(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        try:
            response = self.completions_with_backoff(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            completion = response.choices[0].message.content  # type: ignore
            self.logger.debug("Completion:\n%s", completion)
            completion = json.loads(completion)
            # Turn recipients in messages into ALLCAPS for the engine
            completion["messages"] = {
                recipient.upper(): message
                for recipient, message in completion["messages"].items()
            }
            return ModelResponse(
                model_name=self.model_name,
                reasoning=completion["reasoning"],
                orders=completion["orders"],
                messages=completion["messages"],
            )

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(
                "Error completing prompt ending in\n%s:\n%s",
                user_prompt[:-20],
                exc,
            )
            return ModelResponse(
                model_name=self.model_name,
                reasoning="Error completing prompt.",
                orders=[],
                messages={},
            )

    @backoff.on_exception(backoff.expo, RateLimitError)
    def completions_with_backoff(self, **kwargs):
        """Exponential backoff for OpenAI API rate limit errors."""
        temperature = kwargs.get("temperature", 0.0)
        response = openai.ChatCompletion.create(**kwargs, temperature=temperature)
        assert response is not None, "OpenAI response is None"
        assert "choices" in response, "OpenAI response does not contain choices"
        return response
