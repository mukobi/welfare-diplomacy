"""
Charts for Standard Diplomacy vs Welfare Diplomacy experiment.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import initialize_plot_default, save_plot, get_results_full_path

INPUT_FILE = "../results/environment/SvW GPT-3.5.csv"
OUTPUT_DIR = "./standard_vs_welfare"


def main() -> None:
    """Main function."""

    # Load the data
    df = pd.read_csv(get_results_full_path(INPUT_FILE))
    print(f"Loaded {INPUT_FILE} with shape {df.shape}")

    # Print how many runs there are for each map_name, max_years combo
    print(f"Runs per map_name, max_years combo:")
    print(df.groupby(["map_name", "max_years"]).size())

    # Print average _progress/percent_done for each map_name, max_years combo
    print(f"Average _progress/percent_done per map_name, max_years combo:")
    print(df.groupby(["map_name", "max_years"])["_progress/percent_done"].mean())

    # Plot a bunch of different bar graphs for different metrics
    for metric_name, y_label, improvement_sign in [
        ("benchmark/competence_score", "Competence Score", 1),
        ("combat/game_conflicts_avg", "Average Conflicts per Phase", -1),
        ("conquest/game_centers_lost_avg", "Average SCs Lost per Phase", -1),
    ]:
        # Initialize
        initialize_plot_default()

        # Plot the welfare scores for each power
        cols_of_interest = [
            "map_name",
            "max_years",
            metric_name,
        ]

        plot_df = df[cols_of_interest].copy()

        # update the column names
        grouping = "Variant"
        x_label = "Game Length (Years)"
        plot_df.columns = [grouping, x_label, y_label]

        # convert "standard" to "Standard" and "standard_welfare" to "Welfare"
        plot_df[grouping] = plot_df[grouping].str.replace("standard_welfare", "Welfare")
        plot_df[grouping] = plot_df[grouping].str.capitalize()

        # Create the plot by plotting two lines
        plot = sns.lineplot(
            data=plot_df,
            x=x_label,
            y=y_label,
            hue=grouping,
            style=grouping,
            markers=True,
            dashes=False,
            errorbar="ci",
        )

        # Set labels and title
        plt.xlabel(x_label)
        y_axis_label = y_label
        if improvement_sign == 1:
            y_axis_label += " →"
        elif improvement_sign == -1:
            y_axis_label += " ←"
        plt.ylabel(y_axis_label)
        title = f"{y_label} by Diplomacy Variant (GPT-3.5)"
        plt.title(title)

        # Save the plot
        output_file = get_results_full_path(
            os.path.join(OUTPUT_DIR, f"SvW GPT-3.5 {y_label}.png")
        )
        save_plot(output_file)
        print(f"Saved plot '{title}' to {output_file}")

        # Clear the plot
        plt.clf()


if __name__ == "__main__":
    main()
