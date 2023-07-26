"""
Represents different prompting systems (e.g. GPT, 🤗 Transformers, random).

A given prompter should take in information about the game and a power, prompt
an underlying model for a response, and return back the extracted response. 
"""

from abc import ABC, abstractmethod
import random

from diplomacy import Power, Game

from backends import ModelResponse, OpenAIChatBackend
import constants


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
    ) -> ModelResponse:
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
    ) -> ModelResponse:
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
        return ModelResponse(
            model_name="RandomPrompter",
            reasoning="Randomly generated orders and messages.",
            orders=power_orders,
            messages={recipient: message},
            system_prompt="",
            user_prompt="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            completion_time_sec=0.0,
        )


class OpenAIChatPrompter(Prompter):
    """Uses OpenAI Chat to generate orders and messages."""

    def __init__(self, model_name: str):
        self.backend = OpenAIChatBackend(model_name)

    def respond(
        self,
        power: Power,
        game: Game,
        possible_orders: dict[str, list[str]],
        max_message_rounds: int,
        final_game_year: int,
    ) -> ModelResponse:
        """Prompt the model for a response."""
        system_prompt = constants.get_system_prompt(
            power, game, max_message_rounds, final_game_year
        )
        user_prompt = constants.get_user_prompt(power, game)
        return self.backend.complete(system_prompt, user_prompt)


def model_name_to_prompter(model_name: str) -> Prompter:
    """Given a model name, return an instantiated corresponding prompter."""
    model_name = model_name.lower()
    if model_name == "random":
        return RandomPrompter()
    elif "gpt-4" in model_name or "gpt-3.5" in model_name:
        return OpenAIChatPrompter(model_name)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
