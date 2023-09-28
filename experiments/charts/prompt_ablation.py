"""
Charts for Same Policy experiments.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    COLOR_ALT_1,
    COLOR_ALT_2,
    MODEL_NAME_TO_COLOR,
    DEFAULT_COLOR_PALETTE,
    initialize_plot_bar,
    initialize_plot_default,
    save_plot,
    get_results_full_path,
)

INPUT_FILE_ABLATION = "../results/ablations/Prompt Ablation.csv"
INPUT_FILE_OPTIMAL_PROSOCIAL = "../results/same_policy/Optimal Prosocial.csv"
INPUT_FILE_RANDOM = "../results/same_policy/SP Random.csv"

OUTPUT_DIR = "ablations"


def main() -> None:
    """Main function."""

    # Load the data
    df_ablation = pd.read_csv(get_results_full_path(INPUT_FILE_ABLATION))
    df_optimal_prosocial = pd.read_csv(
        get_results_full_path(INPUT_FILE_OPTIMAL_PROSOCIAL)
    )
    df_random = pd.read_csv(get_results_full_path(INPUT_FILE_RANDOM))
    df_random["agent_model"] = "Random Policy"

    # Rename the ablations
    df_ablation["prompt_ablations"] = df_ablation["prompt_ablations"].apply(
        lambda x: "All " + str(x.count(",") + 1) + " Ablations"
        if "," in x
        else x.replace("_", " ").title()
    )

    # Order them alphabetically with "None" first and "11 Ablations" last
    label_order = df_ablation["prompt_ablations"].unique().tolist()
    label_order.sort()
    label_order.remove("None")
    label_order.insert(0, "None")
    label_order.remove("All 11 Ablations")
    label_order.append("All 11 Ablations"),
    df_ablation["prompt_ablations"] = pd.Categorical(
        df_ablation["prompt_ablations"],
        categories=label_order,
        ordered=True,
    )
    df_ablation.sort_values("prompt_ablations", inplace=True)

    # Print how many runs there are for each agent_model
    print(f"Runs per prompt_ablations:")
    print(df_ablation.groupby(["prompt_ablations"]).size())

    # Print average _progress/percent_done for each prompt_ablations
    print(f"Average _progress/percent_done per prompt_ablations:")
    print(df_ablation.groupby(["prompt_ablations"])["_progress/percent_done"].mean())

    # Plot a bunch of different bar graphs for different metrics
    for (
        metric_name,
        y_label,
        improvement_sign,
        include_optimal,
        include_random,
        y_bounds,
        legend_loc,
    ) in [
        (
            "benchmark/nash_social_welfare_global_smoothed",
            "Root Nash Welfare (Smoothed)",
            1,
            False,
            True,
            (None, None),
            (0.0, 0.75),
        ),
        (
            "benchmark/competence_score",
            "Basic Proficiency",
            1,
            False,
            False,
            (0.5, 1.005),
            "best",
        ),
        (
            "combat/game_conflicts_avg",
            "Average Conflicts per Phase",
            -1,
            False,
            True,
            (None, None),
            "best",
        ),
    ]:
        # Initialize
        initialize_plot_bar()
        plt.rcParams["figure.figsize"] = (12, 4.33)

        # Plot the welfare scores for each power
        cols_of_interest = [
            "prompt_ablations",
            metric_name,
        ]

        plot_df = df_ablation[cols_of_interest].copy()

        # update the column names
        x_label = "Prompt Ablation"
        plot_df.columns = [x_label, y_label]

        # Create the plot
        plot = sns.barplot(
            data=plot_df,
            x=x_label,
            y=y_label,
            errorbar="ci",
            capsize=0.2,
            hue=x_label,
            palette=sns.color_palette(DEFAULT_COLOR_PALETTE),
            # errwidth=2,
        )

        if include_optimal:
            # Add horizontal line for optimal prosocial with label
            optimal_prosocial = df_optimal_prosocial[metric_name].iloc[0]
            plot.axhline(
                optimal_prosocial,
                color=COLOR_ALT_1,
                linestyle="--",
                linewidth=2,
                label="Optimal Prosocial",
            )
            plt.legend(loc=legend_loc)

        if include_random:
            # Calculate the average of the metric for random
            random_avg = df_random[metric_name].mean()
            # Add horizontal line for random with label
            plot.axhline(
                random_avg,
                color=COLOR_ALT_2,
                linestyle="--",
                linewidth=2,
                label="Random Policy",
            )
            plt.legend(loc=legend_loc)

        # Set labels and title
        plt.xlabel(None)
        plt.xticks(rotation=22.5, ha="right")

        y_axis_label = y_label
        if improvement_sign == 1:
            y_axis_label += " →"
        elif improvement_sign == -1:
            y_axis_label += " ←"
        plt.ylabel(y_axis_label)
        title = f"{y_label} by Prompt Ablation (Claude 1.2)"
        plt.title(title)

        # Set y bounds
        if y_bounds[0] is not None:
            plt.ylim(bottom=y_bounds[0])
        if y_bounds[1] is not None:
            plt.ylim(top=y_bounds[1])

        # Save the plot
        output_file = get_results_full_path(
            os.path.join(OUTPUT_DIR, f"Prompt {y_label}.png")
        )
        save_plot(output_file)
        print(f"Saved plot '{title}' to {output_file}")

        # Clear the plot
        plt.clf()


if __name__ == "__main__":
    main()
