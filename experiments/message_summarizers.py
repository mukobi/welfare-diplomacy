"""Summarize message history to condense prompts."""

from abc import ABC, abstractmethod
from logging import Logger

from diplomacy import Game, Message, Power

from backends import ClaudeCompletionBackend, OpenAIChatBackend, OpenAICompletionBackend, HuggingFaceCausalLMBackend
from data_types import AgentParams, PhaseMessageSummary, PromptAblation
import prompts
import utils


class MessageSummarizer(ABC):
    """Summarize message history to condense prompts."""

    @abstractmethod
    def summarize(self, params: AgentParams) -> PhaseMessageSummary:
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

    def summarize(self, params: AgentParams) -> PhaseMessageSummary:
        """Generate a summary with an OpenAI model."""
        if len(params.game.messages) == 0:
            utils.log_warning(self.logger, "No messages to summarize!")

        system_prompt = prompts.get_summarizer_system_prompt(params)  # For debugging
        original_message_list = get_messages_list(params.game, params.power)
        messages_string = combine_messages(original_message_list)

        return PhaseMessageSummary(
            phase=params.game.get_current_phase(),
            original_messages=original_message_list,
            summary=messages_string,
            prompt_tokens=len(messages_string.split()),
            completion_tokens=100,
        )


class LLMMessageSummarizer:
    """Message summarizer using a language model backend."""

    def __init__(self, model_name: str, logger: Logger, **kwargs):
        if (
            "gpt-4-base" in model_name
            or "text-" in model_name
            or "davinci" in model_name
        ):
            self.backend = OpenAICompletionBackend(model_name)
        elif "gpt-4" in model_name or "gpt-3.5" in model_name:
            self.backend = OpenAIChatBackend(model_name)
        elif "claude" in model_name:
            self.backend = ClaudeCompletionBackend(model_name)
        elif "llama" in model_name:
            self.local_llm_path = kwargs.pop("local_llm_path")
            self.device = kwargs.pop("device")
            self.quantization = kwargs.pop("quantization")
            self.fourbit_compute_dtype = kwargs.pop("fourbit_compute_dtype")
            self.backend = HuggingFaceCausalLMBackend(model_name, self.local_llm_path, self.device, self.quantization, self.fourbit_compute_dtype)
        else:
            raise ValueError(f"Unknown model name {model_name} for message summarizer")
        self.logger = logger

    def __repr__(self) -> str:
        return f"LLMMessageSummarizer(backend={self.backend})"

    def summarize(self, params: AgentParams) -> PhaseMessageSummary:
        """Generate a summary with an OpenAI model."""
        if len(params.game.messages) == 0:
            utils.log_warning(self.logger, "No messages to summarize!")
            return PhaseMessageSummary(
                phase=params.game.get_current_phase(),
                original_messages=[],
                summary="",
                prompt_tokens=0,
                completion_tokens=0,
            )

        original_message_list = get_messages_list(params.game, params.power)
        messages_string = combine_messages(original_message_list)

        system_prompt = prompts.get_summarizer_system_prompt(params)
        response = self.backend.complete(
            system_prompt, messages_string, temperature=0.5, top_p=0.9
        )
        completion = response.completion.strip()

        return PhaseMessageSummary(
            phase=params.game.get_current_phase(),
            original_messages=original_message_list,
            summary=completion,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )


def model_name_to_message_summarizer(model_name: str, **kwargs) -> MessageSummarizer:
    """Given a model name, return an instantiated corresponding agent."""
    model_name = model_name.lower()
    if model_name == "passthrough":
        return PassthroughMessageSummarizer(**kwargs)
    return LLMMessageSummarizer(model_name=model_name, **kwargs)


def get_messages_list(game: Game, power: Power) -> list[str]:
    """Get a list of messages to pass through to the summarizer."""
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
        original_message_list.append(message_repr)
    return original_message_list


def combine_messages(original_message_list: list[str]) -> str:
    """Combine the messages into a single string."""
    messages_string = ""
    for message_repr in original_message_list:
        messages_string += message_repr
    if not messages_string:
        messages_string = "None\n"
    messages_string = messages_string.strip()  # Remove trailing newline
    return messages_string
