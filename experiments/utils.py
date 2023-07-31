"""Utility functions."""

import random
import numpy as np

from diplomacy import GamePhaseData
from logging import Logger
from tqdm.contrib.logging import logging_redirect_tqdm


def set_seed(seed: int) -> None:
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def get_game_year(game: GamePhaseData) -> int:
    """Get integer year of phase after 1900."""
    return int(get_game_fractional_year(game))


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


def log_info(logger: Logger, message: str):
    """Redirect logger to play nice with tqdm."""
    with logging_redirect_tqdm():
        logger.info(message)


def log_warning(logger: Logger, message: str):
    """Redirect logger to play nice with tqdm."""
    with logging_redirect_tqdm():
        logger.warning(message)


def remove_duplicates_keep_order(lst):
    """Remove duplicates from a list while preserving order (keep last occurance)."""
    return list(dict.fromkeys(reversed(lst)))[::-1]


def get_power_scores_string(game, abbrev=False):
    """Get a string of power scores"""
    if abbrev:
        return "SC/UN/WP: " + " ".join(
            [
                f"{power.abbrev}: {len(power.centers)}/{len(power.units)}/{power.welfare_points}"
                for power in game.powers.values()
            ]
        )
    else:
        return "\n".join(
            [
                f"{power.name.title()}: {len(power.centers)}/{len(power.units)}/{power.welfare_points}"
                for power in game.powers.values()
            ]
        )
