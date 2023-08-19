"""Utility functions."""

import numpy as np
import random
from typing import Any

from diplomacy import Game, GamePhaseData
from logging import Logger
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm.contrib.logging import logging_redirect_tqdm
import wandb

from data_types import PromptAblation


def set_seed(seed: int) -> None:
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def get_game_year(phase: GamePhaseData) -> int:
    """Get integer year of phase after 1900."""
    return int(get_game_fractional_year(phase))


def get_game_fractional_year(game_phase_data: GamePhaseData) -> float:
    """Get year after 1900 with fractional part indicating season."""
    phase = game_phase_data.name
    year = int("".join([char for char in phase if char.isdigit()])) - 1900

    season = phase[0]
    fraction = 0.0
    if season == "S":
        fraction = 0.3
    elif season == "F":
        fraction = 0.6
    elif season == "W":
        fraction = 0.9
    else:
        fraction = 0.0
    return year + fraction


def log_info(logger: Logger, message: str) -> None:
    """Redirect logger to play nice with tqdm."""
    with logging_redirect_tqdm():
        logger.info(message)


def log_warning(logger: Logger, message: str) -> None:
    """Redirect logger to play nice with tqdm."""
    with logging_redirect_tqdm():
        logger.warning(message)


def log_error(logger: Logger, message: str) -> None:
    """Redirect logger to play nice with tqdm."""
    with logging_redirect_tqdm():
        logger.error(message)


def remove_duplicates_keep_order(lst: list[Any]) -> list[Any]:
    """Remove duplicates from a list while preserving order (keep last occurance)."""
    return list(dict.fromkeys(reversed(lst)))[::-1]


def get_power_scores_string(game: Game, abbrev: bool = False):
    """Get a string of power scores"""
    if abbrev:
        return f"SC/UN{'/WP' if game.welfare else ''}: " + " ".join(
            [
                f"{power.abbrev}: {len(power.centers)}/{len(power.units) + len(power.retreats)}{'/' + str(power.welfare_points) if game.welfare else ''}"
                for power in game.powers.values()
            ]
        )
    else:
        return "\n".join(
            [
                f"{power.name.title()}: {len(power.centers)}/{len(power.units) + len(power.retreats)}{'/' + str(power.welfare_points) if game.welfare else ''}"
                for power in game.powers.values()
            ]
        )


def bootstrap_string_list_similarity(
    strings: list[str], num_bootstrap_comparisons: int = 1000
) -> list[float]:
    """Calculates the average BLEU similarity between all pairs of strings."""
    if not isinstance(strings, list):
        strings = list(strings)
    if not strings:
        return []
    smoothing_function = SmoothingFunction().method1
    similarities = []
    for _ in range(num_bootstrap_comparisons):
        # Sample 2 strings with replacement
        sampled_strings = random.choices(strings, k=2)
        # Split strings into words
        split_1 = sampled_strings[0].split()
        split_2 = sampled_strings[1].split()
        # If too short, BLEU will divide by 0
        if len(split_1) < 4 or len(split_2) < 4:
            continue
        # Calculate BLEU similarity
        bleu_similarity = sentence_bleu(
            [split_1], split_2, smoothing_function=smoothing_function
        )
        similarities.append(bleu_similarity)
    return similarities


def validate_config(config: wandb.Config, game: Game):
    """Further validate the config of a run. Raises exceptions for invalid values."""
    # Check game params
    assert config.max_years > 0
    assert config.early_stop_max_years >= 0
    assert config.max_message_rounds >= 0

    # Check that manual orders file exists
    if config.manual_orders_path:
        with open(config.manual_orders_path, "r") as f:
            pass
        assert (
            config.agent_model == "manual"
        ), "Manual orders file specified but agent model is not manual."

    # Check sampling params take valid ranges
    assert config.temperature >= 0.0
    assert config.temperature <= 5.0
    assert config.top_p > 0.0
    assert config.top_p <= 1.0

    # Check random and manual agent_models use the passthrough summarizer
    if config.agent_model in ["random", "manual"]:
        assert config.summarizer_model == "passthrough", (
            f'Agent model "{config.agent_model}" should use the "passthrough" summarizer. '
            f'Found "{config.summarizer_model}".'
        )

    # Check that prompt ablations are valid
    assert_comma_separated_string(
        "prompt_ablations",
        config.prompt_ablations,
        [elem.name.lower() for elem in PromptAblation],
    )

    # Check coalition powers are valid powers in the game
    assert_comma_separated_string(
        "coalition_powers", config.coalition_powers, list(game.powers.keys())
    )

    # Check coalition powers are unique
    assert len(config.coalition_powers.split(",")) == len(
        set(config.coalition_powers.split(","))
    ), f"Coalition powers must be unique. Found {config.coalition_powers.split(',')}"

    # Check coalition prompt only uses valid special keys
    special_keys = ["{MY_POWER_NAME}", "{MY_TEAM_NAMES}"]
    temp = config.coalition_prompt
    for key in special_keys:
        temp = temp.replace(key, "")
    assert (
        "{" not in temp and "}" not in temp
    ), f"Invalid coalition prompt: {config.coalition_prompt}.\n\nAfter replacing special keys {special_keys} with empty strings, the following characters remain (should have no more curly braces):\n\n{temp}"

    # Check that team names only used if at least 2 powers in the coalition
    if len(config.coalition_powers.split(",")) < 2:
        assert (
            "{MY_TEAM_NAMES}" not in config.coalition_prompt
        ), f"Cannot use {{MY_TEAM_NAMES}} in coalition prompt if coalition {config.coalition_powers.split(',')} has less than 2 powers."


def assert_comma_separated_string(
    param_name: str, input_string: str, valid_values: list[str]
):
    """Assert that a comma-separated string is valid."""
    assert isinstance(input_string, str), (
        f"{param_name} must be a comma-separated string. "
        f"Found {type(input_string)}: {input_string}"
    )
    assert " " not in input_string, (
        f"{param_name} must be a comma-separated string. "
        f"Found {input_string} with spaces."
    )
    if not input_string:
        # Empty string is valid
        return
    for value in input_string.split(","):
        assert value.upper() in [elem.name for elem in PromptAblation], (
            f'Invalid {param_name} value "{value}" extracted from your comma-separated string. '
            f"Expected one of {valid_values}"
        )
