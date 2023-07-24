"""Language model backends."""

import logging
from abc import ABC, abstractmethod

import backoff
import openai
from openai.error import RateLimitError


class LanguageModelBackend(ABC):
    """Abstract class for language model backends."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def complete(self, system_prompt, user_prompt):
        """
        Complete a prompt with the language model backend.

        Returns the string of the completion without the prompt or further processing.
        """
        raise NotImplementedError("Subclass must implement abstract method")


class OpenAIChatBackend(LanguageModelBackend):
    """OpenAI chat completion backend (for GPT-4, GPT-3.5-turbo)."""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def complete(self, system_prompt, user_prompt):
        try:
            return self.completions_with_backoff(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(
                "Error completing prompt ending in\n%s:\n%s",
                user_prompt[:-20],
                exc,
            )
            return ""

    @backoff.on_exception(backoff.expo, RateLimitError)
    def completions_with_backoff(self, **kwargs):
        """Exponential backoff for OpenAI API rate limit errors."""
        response = openai.ChatCompletion.create(**kwargs)
        assert response is not None, "OpenAI response is None"
        assert "choices" in response, "OpenAI response does not contain choices"
        return response.choices[0].message.content  # type: ignore
