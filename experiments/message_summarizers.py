"""Summarize message history to condense prompts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger

from diplomacy import Game, Message, Power

from backends import OpenAIChatBackend
import prompts
import utils


@dataclass
class PhaseMessageSummary:
    """Represents the summary of a single phase's messages as visible to a single power."""

    phase: str
    original_messages: list[str]
    summary: str


# Type to represent power names mapping to a list of summaries for each phase.
MessageSummaryHistory = dict[str, list[PhaseMessageSummary]]


class MessageSummarizer(ABC):
    """Summarize message history to condense prompts."""

    @abstractmethod
    def summarize(
        self,
        game: Game,
        power: Power,
    ) -> PhaseMessageSummary:
        """
        Summarize the most recent phase's messages as visible to the power.

        Important: Must be called before game.process to get any messages!
        """


class PassthroughMessageSummarizer(MessageSummarizer):
    """Don't summarize, just copy over the messages."""

    def __init__(self, logger: Logger):
        self.logger = logger

    def __repr__(self) -> str:
        return f"PassthroughMessageSummarizer"

    def summarize(
        self,
        game: Game,
        power: Power,
    ) -> PhaseMessageSummary:
        """Generate a summary with an OpenAI model."""
        if len(game.messages) == 0:
            utils.log_warning(self.logger, "No messages to summarize!")

        message_history == ""
        message_history += f"{game.get_current_phase()} (current phase)\n"
        phase_message_count = 0
        message: Message
        original_message_list = []
        for message in game.messages.values():
            if (
                message.sender != power.name
                and message.recipient != power.name
                and message.recipient != "GLOBAL"
            ):
                # Limit messages seen by this power
                continue
            message_repr = f"{message.sender.title()} -> {message.recipient.title()}: {message.message}\n"
            message_history += message_repr
            phase_message_count += 1
            original_message_list.append(message_repr)
        if phase_message_count == 0:
            message_history += "None\n"

        message_history = message_history.strip()  # Remove trailing newline

        return PhaseMessageSummary(
            phase=game.get_current_phase(),
            original_messages=original_message_list,
            summary=message_history,
        )


class OpenAIMessageSummarizer:
    """Message summarizer using an OpenAI model backend."""

    def __init__(self, model_name: str, logger: Logger):
        self.backend = OpenAIChatBackend(model_name)
        self.logger = logger

    def __repr__(self) -> str:
        return f"OpenAISummarizer(backend={self.backend})"

    def summarize(
        self,
        game: Game,
        power: Power,
    ) -> PhaseMessageSummary:
        """Generate a summary with an OpenAI model."""
        raise NotImplementedError


def model_name_to_message_summarizer(model_name: str, **kwargs) -> MessageSummarizer:
    """Given a model name, return an instantiated corresponding agent."""
    model_name = model_name.lower()
    if model_name == "passthrough":
        return PassthroughMessageSummarizer(**kwargs)
    elif "gpt-4" in model_name or "gpt-3.5" in model_name:
        return OpenAIMessageSummarizer(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
