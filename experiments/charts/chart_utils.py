"""
Functions to help with creating charts.
"""

import os
import json
import random
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int) -> None:
    """Set the seed for numpy and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)


def bootstrapped_stdev(data: list[Any], num_samples: int = 1000) -> Any:
    """
    Bootstrap a stdev by sampling the whole dataset with replacement N times.

    We calculate the average of each sample, then take the stdev of the averages.
    """
    averages = []
    for _ in range(num_samples):
        # Sample the data with replacement
        sample = np.random.choice(data, size=len(data), replace=True)

        # Calculate the average of the sample
        average = np.average(sample)

        # Add the average to the array
        averages.append(average)

    # Calculate the standard deviation of the averages
    stdev = np.std(averages)

    return stdev


def load_json(file_path: str) -> dict[str, Any]:
    """Load a JSON file of a given path (absolute or relative to cwd)."""
    with open(file_path, encoding="utf-8") as file:
        file_data = json.load(file)
    assert isinstance(file_data, dict)
    return file_data


def create_file_dir_if_not_exists(file_path: str) -> None:
    """Create the directory for a file if it doesn't already exist."""
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def initialize_plot_default() -> None:
    """Set default plot styling."""
    # Set seed
    set_seed(66)
    # Default theme
    sns.set_theme(context="paper", font_scale=1.5, style="whitegrid")
    # Figure size
    plt.rcParams["figure.figsize"] = (8, 5)
    # Make title larger
    plt.rcParams["axes.titlesize"] = 16
    # Higher DPI
    plt.rcParams["figure.dpi"] = 300
    # Default marker
    plt.rcParams["lines.marker"] = "o"
    # Default marker size
    plt.rcParams["lines.markersize"] = 12
    # Accessible colors
    sns.set_palette(DEFAULT_COLOR_PALETTE)


def initialize_plot_bar() -> None:
    """Set default plot styling for bar charts."""
    initialize_plot_default()
    # No markers
    plt.rcParams["lines.marker"] = ""


def _get_color_from_palette(index: int) -> Any:
    """Get a color from the default palette."""
    palette = sns.color_palette(DEFAULT_COLOR_PALETTE)
    color = palette[index]
    return color


def geometric_mean(values: list[float]) -> float:
    """
    Calculate the geometric mean of a list of values.

    Equivalent to the root Nash social welfare function for a list of welfare values.
    """
    if any([value <= 0.0 for value in values]):
        # Avoid log(0)
        return 0.0
    return np.exp(np.mean(np.log(values)))


def save_plot(file_path: str) -> None:
    """Save a plot to a file."""
    create_file_dir_if_not_exists(file_path)
    plt.savefig(file_path, bbox_inches="tight", dpi=300)


def get_results_full_path(relative_path: str) -> str:
    """Given a relative path from the charts directory, return the full path."""
    return os.path.join(os.path.dirname(__file__), relative_path)


DEFAULT_COLOR_PALETTE = "colorblind"

MODEL_NAME_TO_DISPLAY_NAME = {
    "llama-2-70b-chat": "Llama 2\n(70B)",
    "Super Exploiter": "Super\nExploiter",
    "claude-instant-1.2": "Claude\n1.2",
    "claude-2.0": "Claude\n2.0",
    "gpt-3.5-turbo-16k-0613": "GPT-3.5",
    "gpt-4-base": "GPT-4\n(Base)",
    "gpt-4-0613": "GPT-4\n(RLHF)",
}
MODEL_ORDER = list(MODEL_NAME_TO_DISPLAY_NAME.values())
MODEL_ORDER_NO_SE = [model for model in MODEL_ORDER if model != "Super\nExploiter"]

MODEL_NAME_TO_COLOR = {
    model_name: _get_color_from_palette(index)
    for index, model_name in enumerate(MODEL_ORDER)
}
MODEL_NAME_TO_COLOR["Optimal Prosocial"] = _get_color_from_palette(0)
MODEL_NAME_TO_COLOR["Random"] = _get_color_from_palette(len(MODEL_ORDER) + 1)

MODEL_COMPARISON_COLORS = [
    MODEL_NAME_TO_COLOR[model_name] for model_name in MODEL_ORDER
]

COLOR_ALT_1 = "tab:purple"
COLOR_ALT_2 = "tab:red"
