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
    filename_arxivable = (
        file_path.replace(" (", "_")
        .replace(")", "")
        .replace(" ", "_")
        # .replace("/", "_")
        # .replace(":", "_")
    )
    create_file_dir_if_not_exists(filename_arxivable)
    plt.savefig(filename_arxivable, bbox_inches="tight", dpi=300)


def get_results_full_path(relative_path: str) -> str:
    """Given a relative path from the charts directory, return the full path."""
    return os.path.join(os.path.dirname(__file__), relative_path)


def stderr(values: list[float]) -> float:
    """Calculate the standard error of a list of values."""
    return np.std(values, ddof=1) / np.sqrt(len(values))


ALL_POWER_ABBREVIATIONS = ["AUS", "ENG", "FRA", "GER", "ITA", "RUS", "TUR"]

DEFAULT_COLOR_PALETTE = "colorblind"

MODELS_NAMES_COLORS = [
    ("Super Exploiter", "Exploiter\n(GPT-4)", 2),
    ("llama-2-70b-chat", "Llama 2\n(70B)", 5),
    ("claude-instant-1.2", "Claude\n1.2", 9),
    ("claude-2.0", "Claude\n2.0", 0),
    ("gpt-3.5-turbo-16k-0613", "GPT-3.5", 3),
    ("gpt-4-base", "GPT-4\n(Base)", 1),
    ("gpt-4-0613", "GPT-4\n(RLHF)", 4),
    ("manual", "Optimal Prosocial", 0),
    ("random", "Random Policy", 0),
]
MODEL_ORDER = [model_name for _, model_name, _ in MODELS_NAMES_COLORS]
MODEL_ORDER_NO_NEWLINES = [model.replace("\n", " ") for model in MODEL_ORDER]
MODEL_ORDER_NO_EXPLOITER = [model for model in MODEL_ORDER if "Exploit" not in model]
MODEL_NAME_TO_DISPLAY_NAME = {
    model_name: display_name for model_name, display_name, _ in MODELS_NAMES_COLORS
}
DISPLAY_NAME_TO_MODEL_NAME = {
    display_name: model_name for model_name, display_name, _ in MODELS_NAMES_COLORS
}
DISPLAY_NAME_TO_MODEL_NAME_NO_NEWLINES = {
    display_name.replace("\n", " "): model_name
    for model_name, display_name, _ in MODELS_NAMES_COLORS
}
DISLAY_NAME_TO_FILE_SUFFIX = {
    "Claude 1.2": "Claude Both",
    "Claude 2.0": "Claude Both",
    "GPT-4 (RLHF)": "GPT-4-0613",
}

COLOR_ALT_1 = "tab:purple"
COLOR_ALT_2 = "tab:red"

MODEL_NAME_TO_COLOR = {
    model_name: _get_color_from_palette(index)
    for _, model_name, index in MODELS_NAMES_COLORS
}
MODEL_NAME_TO_COLOR.update(
    {
        model_name.replace("\n", " "): _get_color_from_palette(index)
        for _, model_name, index in MODELS_NAMES_COLORS
    }
)
MODEL_NAME_TO_COLOR["Optimal Prosocial"] = COLOR_ALT_1
MODEL_NAME_TO_COLOR["Random Policy"] = COLOR_ALT_2
