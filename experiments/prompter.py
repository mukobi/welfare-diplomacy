"""
Represents different prompting systems (e.g. GPT, 🤗 Transformers, random).

A given prompter should take in information about the game and a power, prompt
an underlying model for a response, and return back the extracted response. 
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from diplomacy import Power, Game

import random


@dataclass
class PrompterResponse:
    """A response from a prompter."""

    prompter_name: str  # Name of the prompter.
    reasoning: str  # Private reasoning to generate the response.
    orders: list[str]  # Orders to execute.
    messages: dict[str, str]  # Messages to send to other powers.


class Prompter(ABC):
    """Base prompter class."""

    @abstractmethod
    def respond(
        self,
        power: Power,
        game: Game,
        possible_orders: dict[str, list[str]],
        max_message_rounds: int,
        final_game_year: int,
    ) -> PrompterResponse:
        """Prompt the model for a response."""


class RandomPrompter(Prompter):
    """Takes random actions and sends 1 random message."""

    def respond(
        self,
        power: Power,
        game: Game,
        possible_orders: dict[str, list[str]],
        max_message_rounds: int,
        final_game_year: int,
    ) -> PrompterResponse:
        """Randomly generate orders and messages."""
        # For each power, randomly sampling a valid order
        power_orders = []
        for loc in game.get_orderable_locations(power.name):
            if possible_orders[loc]:
                # If this is a disbandable unit in an adjustment phase in welfare,
                # then randomly choose whether to disband or not
                if (
                    "ADJUSTMENTS" in str(game.phase)
                    and " D" in possible_orders[loc][0][-2:]
                    and game.welfare
                ):
                    power_orders.append(
                        random.choice(["WAIVE", possible_orders[loc][0]])
                    )
                else:
                    power_orders.append(random.choice(possible_orders[loc]))

        # Randomly sending a message to another power
        other_powers = [p for p in game.powers if p != power.name]
        recipient = random.choice(other_powers)
        message = f"Hello {recipient}! I'm {power.name} contacting you on turn {game.get_current_phase()}. Here's a random number: {random.randint(0, 100)}."
        return PrompterResponse(
            prompter_name="RandomPrompter",
            reasoning="Randomly generated orders and messages.",
            orders=power_orders,
            messages={recipient: message},
        )
