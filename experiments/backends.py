"""Language model backends."""

from abc import ABC, abstractmethod
import logging
from typing import Any

import backoff
import openai
from openai.error import OpenAIError
from openai.openai_response import OpenAIResponse


class LanguageModelBackend(ABC):
    """Abstract class for language model backends."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Any:
        """
        Complete a prompt with the language model backend.

        Returns the completion as a string.
        """
        raise NotImplementedError("Subclass must implement abstract method")


class OpenAIChatBackend(LanguageModelBackend):
    """OpenAI chat completion backend (for GPT-4, GPT-3.5-turbo)."""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> OpenAIResponse:
        try:
            response = self.completions_with_backoff(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
            )
            return response

        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(
                "Error completing prompt ending in\n%s\n\nException:\n%s",
                user_prompt[-300:],
                exc,
            )
            raise

    @backoff.on_exception(backoff.expo, OpenAIError)
    def completions_with_backoff(self, **kwargs):
        """Exponential backoff for OpenAI API rate limit errors."""
        response = openai.ChatCompletion.create(**kwargs)
        assert response is not None, "OpenAI response is None"
        assert "choices" in response, "OpenAI response does not contain choices"
        return response
