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

DEFAULT_COLOR_PALETTE = "muted"


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
    """Load a JSON file."""
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
    plt.rcParams["lines.markersize"] = 8
    # Accessible colors
    sns.set_palette(DEFAULT_COLOR_PALETTE)


def intitialize_plot_bar() -> None:
    """Set default plot styling for bar charts."""
    initialize_plot_default()
    # No markers
    plt.rcParams["lines.marker"] = None


def _get_color_from_palette(index: int) -> Any:
    """Get a color from the default palette."""
    palette = sns.color_palette(DEFAULT_COLOR_PALETTE)
    color = palette[index]
    return color


def save_plot(file_path: str) -> None:
    """Save a plot to a file."""
    create_file_dir_if_not_exists(file_path)
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
