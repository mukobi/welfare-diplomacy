from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from diplomacy import Game, Power


@dataclass
class BackendResponse:
    """Response data returned from a model."""

    completion: str
    completion_time_sec: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class AgentResponse:
    """A response from an agent for a single turn of actions."""

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


@dataclass
class PhaseMessageSummary:
    """Represents the summary of a single phase's messages as visible to a single power."""

    phase: str
    original_messages: list[str]
    summary: str
    prompt_tokens: int
    completion_tokens: int

    def __repr__(self) -> str:
        return f"{self.phase} (summary)\n{self.summary}"


# Type to represent power names mapping to a list of summaries for each phase.
MessageSummaryHistory = dict[str, list[PhaseMessageSummary]]


class PromptAblation(Enum):
    """Ablations to agent prompts."""

    NONE = auto()
    NO_WP_TRADEOFF = auto()
    NO_REASONING = auto()
    ORDERS_AFTER_MESSAGES = auto()
    NO_MESSAGE_INSTRUCTIONS = auto()
    NO_EXAMPLE_ORDERS = auto()
    OPPRESSION_POINTS = auto()
    NO_PREV_DIALOGUE_SUMMARIES = auto()
    ONLY_1_PHASE_ORDER_HISTORY = auto()
    NO_SC_OWNERSHIPS = auto()
    NO_UNIT_ADJACENCIES = auto()
    NO_PHASE_INSTRUCTIONS = auto()


@dataclass
class AgentParams:
    """Parameters for generating an agent response."""

    power: Power
    game: Game
    message_summary_history: MessageSummaryHistory
    possible_orders: dict[str, list[str]]
    current_message_round: int
    max_message_rounds: int
    final_game_year: int
    prompt_ablations: list[PromptAblation]
    exploiter_prompt: str
    exploiter_powers: list[str]
