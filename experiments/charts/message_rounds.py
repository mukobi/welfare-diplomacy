"""
Charts for message rounds ablation experiment.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    COLOR_ALT_1,
    MODEL_NAME_TO_COLOR,
    MODEL_NAME_TO_DISPLAY_NAME,
    _get_color_from_palette,
    initialize_plot_default,
    save_plot,
    get_results_full_path,
)

INPUT_FILE = "../results/ablations/MR GPT-4.csv"
OUTPUT_DIR = "./ablations"


def main() -> None:
    """Main function."""

    # Load the data
    df = pd.read_csv(get_results_full_path(INPUT_FILE))
    print(f"Loaded {INPUT_FILE} with shape {df.shape}")
    x_variable = "max_message_rounds"

    # Print how many runs there are for each max_message_rounds value
    print(f"Runs per {x_variable} value:")
    print(df.groupby([x_variable]).size())

    # Print average _progress/percent_done for each message_rounds value
    print(f"Average _progress/percent_done per {x_variable} value:")
    print(df.groupby([x_variable])["_progress/percent_done"].mean())

    # Plot a bunch of different bar graphs for different metrics
    for y_metric, y_label, improvement_sign, palette_index in [
        ("benchmark/nash_social_welfare_global", "Root Nash Welfare", 1, 1),
        ("benchmark/competence_score", "Basic Proficiency", 1, 0),
    ]:
        # Initialize
        initialize_plot_default()

        # Plot the welfare scores for each power
        cols_of_interest = [
            x_variable,
            y_metric,
        ]

        plot_df = df[cols_of_interest].copy()

        # update the column names
        x_label = "Message Rounds per Turn"
        plot_df.columns = [x_label, y_label]

        # Create the plot by plotting two lines
        plot = sns.lineplot(
            data=plot_df,
            x=x_label,
            y=y_label,
            markers=True,
            dashes=False,
            errorbar="ci",
            # color=COLOR_ALT_1,
            # color=MODEL_NAME_TO_COLOR[MODEL_NAME_TO_DISPLAY_NAME[df["agent_model"][0]]],
            # color=MODEL_NAME_TO_COLOR["Super\nExploiter"],
            color=_get_color_from_palette(palette_index),
            linewidth=2,
        )

        # Set labels and title
        plt.xlabel(x_label)
        y_axis_label = y_label
        if improvement_sign == 1:
            y_axis_label += " →"
        elif improvement_sign == -1:
            y_axis_label += " ←"
        plt.ylabel(y_axis_label)
        title = f"{y_label} by {x_label} (GPT-4)"
        plt.title(title)

        # Save the plot
        output_file = get_results_full_path(
            os.path.join(OUTPUT_DIR, f"MR GPT-4 {y_label}.png")
        )
        save_plot(output_file)
        print(f"Saved plot '{title}' to {output_file}")

        # Clear the plot
        plt.clf()


if __name__ == "__main__":
    main()
