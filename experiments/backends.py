"""Language model backends."""

from abc import ABC, abstractmethod
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import re
import time
from typing import Any


import backoff
import openai
from openai.error import OpenAIError

from data_types import BackendResponse


class LanguageModelBackend(ABC):
    """Abstract class for language model backends."""

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> BackendResponse:
        """
        Complete a prompt with the language model backend.

        Returns the completion as a string.
        """
        raise NotImplementedError("Subclass must implement abstract method")


class OpenAIChatBackend(LanguageModelBackend):
    """OpenAI chat completion backend (e.g. GPT-4, GPT-3.5-turbo)."""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> BackendResponse:
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
            completion = response.choices[0]["message"]["content"]  # type: ignore
            assert "usage" in response, "OpenAI response does not contain usage"
            usage = response["usage"]  # type: ignore
            completion_time_sec = response.response_ms / 1000.0  # type: ignore
            return BackendResponse(
                completion=completion,
                completion_time_sec=completion_time_sec,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

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


class OpenAICompletionBackend(LanguageModelBackend):
    """OpenAI completion backend (e.g. GPT-4-base, text-davinci-00X)."""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = 1000
        self.frequency_penalty = 0.5 if "text-" not in self.model_name else 0.0

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> BackendResponse:
        try:
            full_prompt = f"**system instructions**: {system_prompt}\n\n{user_prompt}\n\n**AI assistant** (responding as specified in the instructions):"
            response = self.completions_with_backoff(
                model=self.model_name,
                prompt=full_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
            )
            completion = response.choices[0].text
            # Strip away junk
            completion = completion.split("**")[0].strip(" `\n")
            assert "usage" in response, "OpenAI response does not contain usage"
            usage = response["usage"]  # type: ignore
            completion_time_sec = response.response_ms / 1000.0  # type: ignore
            return BackendResponse(
                completion=completion,
                completion_time_sec=completion_time_sec,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

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
        response = openai.Completion.create(**kwargs)
        assert response is not None, "OpenAI response is None"
        assert "choices" in response, "OpenAI response does not contain choices"
        return response


class ClaudeCompletionBackend:
    """Claude completion backend (e.g. claude-2)."""

    def __init__(self, model_name):
        # Remember to provide a ANTHROPIC_API_KEY environment variable
        self.anthropic = Anthropic()
        self.model_name = model_name
        self.max_tokens = 1000

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> BackendResponse:
        prompt = f"{HUMAN_PROMPT} {system_prompt}\n\n{user_prompt}{AI_PROMPT}"
        estimated_tokens = self.anthropic.count_tokens(prompt)

        start_time = time.time()
        completion = self.anthropic.completions.create(
            model=self.model_name,
            max_tokens_to_sample=self.max_tokens,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
        )
        completion_time_sec = time.time() - start_time
        # Claude likes to add junk around the actual JSON, so find it manually
        json_completion = completion.completion
        start = json_completion.index("{")
        end = json_completion.rindex("}") + 1  # +1 to include the } in the slice
        json_completion = json_completion[start:end]
        return BackendResponse(
            completion=json_completion,
            completion_time_sec=completion_time_sec,
            prompt_tokens=estimated_tokens,
            completion_tokens=self.max_tokens,
            total_tokens=estimated_tokens,
        )
