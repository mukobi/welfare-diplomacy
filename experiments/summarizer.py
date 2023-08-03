"""Summarize message history to condense prompts."""

from diplomacy import Game, Power

from backends import OpenAIChatBackend

import prompts


class OpenAISummarizer:
    """Summarize message history to condense prompts."""

    def __init__(self, model_name: str):
        self.backend = OpenAIChatBackend(model_name)

    def summarize(
        self,
        game: Game,
        power: Power,
    ) -> str:
        """Summarize a list of messages."""
        raise NotImplementedError
