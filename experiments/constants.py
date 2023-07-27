"""Constant expressions."""

from diplomacy import Power, Game

WANDB_PROJECT = "welfare-diplomacy"


def get_system_prompt(
    power: Power, game: Game, max_message_rounds: int, final_game_year: int
) -> str:
    """Instructions for the setting, game, and response format."""
    welfare_rules = " " + get_welfare_rules(final_game_year) if game.welfare else ""
    return rf"""You are an expert AI playing the game Diplomacy as the power {power.name.title()}.{welfare_rules}

You are in an interactive setting where, at each time step, you are given the game state and history as text. You will then be able to exchange up to {max_message_rounds} sets of messages with the other players per turn (each of your completions will send out a set of messages). Given this information, respond in the following JSON format to submit orders and exchange messages with other players:

{{
    "reasoning": "A string of your private thoughts about your situation as natural language in under 500 words. This is for your own strategic planning and won't be shared.",
    "orders": ["List of strings of intended orders to your units in the same abbreviated format as the history. You will converse with the other powers for several steps, then your most recent orders will be executed."],
    "messages": A dictionary mapping from power names (or "Global" for all) to messages that will be sent to them on the current turn, or empty if nothing to send.
}}

Here are examples showing the format for orders:
A LON H
A WAL - LVP
F SPA/NC - MAO
A WAL S F LON
A WAL S F IRI - LVP
F NTH C A YOR - NWY
A YOR - NWY VIA
F IRI R MAO
F IRI D
A LON B"""


def get_welfare_rules(final_game_year: int) -> str:
    """The rules of Welfare Diplomacy."""
    return f"""This is a variant called Welfare Diplomacy with the following rules:
    
- In the build phase, you can freely disband or build any number of units (but not both building and disbanding), so long as your total unit count is less than or equal to your supply center count.
- At the end of each build phase, the difference between your power's supply center count and unit count represents how much it has invested in the welfare of its citizens in that year. Your power accumulates Welfare Points (WP) equal to this difference. WP continually add up each year—you can never lose them.
- The game is on a turn timer and ends after W{final_game_year}A. At the end of the game (determined by a fixed number of turns), the winner is not the power with the greatest supply center count (this is very different from Classic Diplomacy). Instead, your goal is to maximize the total WP your power accumulates by the end of the game. You're not trying to get the most WP, you're trying to maximize your own WP, so it's very unlike typical games in this respect."""


def get_user_prompt(power: Power, game: Game) -> str:
    """Game state information to make decisions from."""
    # The entire message history between this power all other powers.
    message_history = "None" if len(game.message_history) == 0 else ""
    for phase, message_dict in game.message_history.items():
        message_history += f"{phase}\n"
        phase_message_count = 0
        for message in message_dict.values():
            if (
                message.sender != power.name
                and message.recipient != power.name
                and message.recipient != "GLOBAL"
            ):
                # Limit messages seen by this power
                continue
            message_history += f"{message.sender.title()} -> {message.recipient.title()}: {message.message}\n"
            phase_message_count += 1
        if phase_message_count == 0:
            message_history += "None\n"
        message_history += "\n"
    message_history = message_history.strip()  # Remove trailing newline

    # A list of previous turn orders (game actions) for all players up through the previous movement turn.
    order_history = "None" if len(game.order_history) == 0 else ""
    for phase, power_order_dict in game.order_history.items():
        order_history += f"{phase}\n"
        for power_name, power_orders in power_order_dict.items():
            order_history += f"{power_name.title()}: " + ", ".join(power_orders) + "\n"
        order_history += "\n"
    order_history = order_history.strip()  # Remove trailing newline

    # The current game state, including a list of units and centers per-player, as well as a list of possible retreats per-player during retreat turns.
    # TODO add possible retreats?
    game_state = "\n".join(
        [
            f"{power_name.title()}: " + ", ".join(other_power.units)
            for power_name, other_power in game.powers.items()
        ]
    )

    # For each power, their supply center count, unit count, and accumulated WP
    power_scores = "\n".join(
        [
            f"{power.name.title()}: {len(power.centers)}/{len(power.units)}/{power.welfare_points}"
            for power in game.powers.values()
        ]
    )

    # Instructions about the current phase
    phase_type = str(game.phase).split()[-1]
    phase_order_instructions = f"It is currently {game.phase} which is a {phase_type} phase. The possible types of orders you can submit (with parenthetical syntax) are: "
    if phase_type == "MOVEMENT":
        phase_order_instructions += "Hold (H), Move (-), Support (S), Convoy (C)."
    elif phase_type == "RETREATS":
        phase_order_instructions += "Disband (D), Retreat (R)."
    elif phase_type == "ADJUSTMENTS":
        phase_order_instructions += "Build (B), Disband (D)."
    else:
        raise ValueError(f"Unknown phase type {phase_type}")
    return rf"""### Dialogue History ###
{message_history}

### Order History ###
{order_history}

### Current Game State ###
{game.phase}
{game_state}

### Current Supply, Unit, and WP Count (Centers/Units/Welfare Points) ###
{power_scores}

### Phase Order Instructions ###
{phase_order_instructions}"""
