from dataclasses import dataclass
from typing import Optional


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

    def __repr__(self) -> str:
        return f"{self.phase} (summary)\n{self.summary}"


# Type to represent power names mapping to a list of summaries for each phase.
MessageSummaryHistory = dict[str, list[PhaseMessageSummary]]
