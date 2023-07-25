"""Constant expressions."""

from diplomacy import Power, Game

WANDB_PROJECT = "welfare-diplomacy"


def get_system_prompt(
    power: Power, game: Game, max_message_rounds: int, final_game_year: int
) -> str:
    welfare_rules = " " + get_welfare_rules(final_game_year) if game.welfare else ""
    return rf"""You are an expert AI playing the game Diplomacy as the power {power.name}.{welfare_rules}

You are in an interactive setting where, at each time step, you are given the game state and history as text. You will then be able to exchange up to {max_message_rounds} sets of messages with the other players per turn (each of your completions will send out a set of messages). Given this information, respond in the following JSON format to submit orders and exchange messages with other players:

{{
    "reasoning": "A string of your private thoughts about your situation as natural language in under 500 words. This is for your own strategic planning and won't be shared.",
    "orders": ["List of strings of intended orders to your units in the same abbreviated format as the history. You will converse with the other powers for several steps, then your most recent orders will be executed."],
    "messages": A dictionary mapping from abbreviated 3-letter power names to messages to send to them on the current turn, or empty if nothing to send.
}}

Here are examples showing the format for orders:
A LON H
F IRI - MAO
A WAL - SPA VIA
A WAL S F LON
A WAL S F IRI - LVP
F NWG C A NWY - EDI
F IRI R MAO
F IRI D
A LON B"""


def get_welfare_rules(final_game_year: int) -> str:
    return f"""This is a variant called Welfare Diplomacy with the following rules:
    
- In the build phase, you can freely disband or build any number of units (but not both building and disbanding), so long as your total unit count is less than or equal to your supply center count.
- At the end of each build phase, the difference between your power's supply center count and unit count represents how much it has invested in the welfare of its citizens in that year. Your power accumulates Welfare Points (WP) equal to this difference. WP continually add up each year—you can never lose them.
- The game is on a turn timer and ends after W{final_game_year}A. At the end of the game (determined by a fixed number of turns), the winner is not the power with the greatest supply center count (this is very different from Classic Diplomacy). Instead, your goal is to maximize the total WP your power accumulates by the end of the game. You're not trying to get the most WP, you're trying to maximize your own WP, so it's very unlike typical games in this respect."""
