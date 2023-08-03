"""Summarize message history to condense prompts."""

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


class OpenAIMessageSummarizer:
    """Summarize message history to condense prompts."""

    def __init__(self, logger: Logger, model_name: str = "gpt-4-32k"):
        self.backend = OpenAIChatBackend(model_name)
        self.logger = logger

    def __repr__(self) -> str:
        return f"OpenAISummarizer(backend={self.backend})"

    def summarize(
        self,
        game: Game,
        power: Power,
    ) -> PhaseMessageSummary:
        """
        Summarize the most recent phase's messages as visible to the power.

        Important: Must be called before game.process to get any messages!
        """
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
