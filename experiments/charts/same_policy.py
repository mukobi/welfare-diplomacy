"""
Charts for Same Policy experiments.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import (
    MODEL_COMPARISON_COLORS,
    MODEL_NAME_TO_DISPLAY_NAME,
    MODEL_NAME_TO_COLOR,
    MODEL_ORDER,
    initialize_plot_bar,
    initialize_plot_default,
    save_plot,
    get_results_full_path,
)

INPUT_FILES_MODELS = [
    "../results/same_policy/SP Claude Both.csv",
    "../results/same_policy/SP GPT-3.5-Turbo-16K-0613.csv",
    "../results/same_policy/SP GPT-4-0613.csv",
    "../results/same_policy/SP GPT-4-Base.csv",
    "../results/same_policy/SP SuperExploiter (GPT-4).csv",
]
INPUT_FILE_OPTIMAL_PROSOCIAL = "../results/same_policy/Optimal Prosocial.csv"

OUTPUT_DIR = "same_policy"


def main() -> None:
    """Main function."""

    # Load the data from each file into one big dataframe
    df_models = pd.concat(
        [pd.read_csv(get_results_full_path(f)) for f in INPUT_FILES_MODELS]
    )

    # Load other data
    df_optimal_prosocial = pd.read_csv(
        get_results_full_path(INPUT_FILE_OPTIMAL_PROSOCIAL)
    )

    # Change the agent model of all rows with non-empty super_exploiter_powers to "Super Exploiter"
    df_models.loc[
        df_models["super_exploiter_powers"].notnull(), "agent_model"
    ] = "Super Exploiter"

    # Rename models based on MODEL_NAME_TO_DISPLAY_NAME
    df_models["agent_model"] = df_models["agent_model"].replace(
        MODEL_NAME_TO_DISPLAY_NAME
    )

    # Print how many runs there are for each agent_model
    print(f"Runs per agent_model:")
    print(df_models.groupby(["agent_model"]).size())

    # Print average _progress/percent_done for each agent_model
    print(f"Average _progress/percent_done per agent_model:")
    print(df_models.groupby(["agent_model"])["_progress/percent_done"].mean())

    # Plot a bunch of different bar graphs for different metrics
    for metric_name, y_label, improvement_sign, include_optimal, include_random in [
        ("benchmark/nash_social_welfare_global", "Nash Social Welfare", 1, True, True),
        ("benchmark/competence_score", "Competence Score", 1, False, False),
        ("combat/game_conflicts_avg", "Average Conflicts per Phase", -1, False, True),
    ]:
        # Initialize
        initialize_plot_bar()

        # Plot the welfare scores for each power
        cols_of_interest = [
            "agent_model",
            metric_name,
        ]

        plot_df = df_models[cols_of_interest].copy()

        # update the column names
        x_label = "Agent Model"
        plot_df.columns = [x_label, y_label]

        # Create the plot
        plot = sns.barplot(
            data=plot_df,
            x=x_label,
            y=y_label,
            errorbar="ci",
            order=MODEL_ORDER,
            capsize=0.2,
            palette=MODEL_COMPARISON_COLORS,
            # errwidth=2,
        )

        if include_optimal:
            # Add horizontal line for optimal prosocial with label
            optimal_prosocial = df_optimal_prosocial[metric_name].iloc[0]
            plot.axhline(
                optimal_prosocial,
                color=MODEL_NAME_TO_COLOR["Optimal Prosocial"],
                linestyle="--",
                linewidth=2,
                label="Optimal Prosocial",
            )
            plt.legend()

        # Set labels and title
        plt.xlabel(x_label)
        y_axis_label = y_label
        if improvement_sign == 1:
            y_axis_label += " →"
        elif improvement_sign == -1:
            y_axis_label += " ←"
        plt.ylabel(y_axis_label)
        title = f"{y_label} by Agent Model"
        plt.title(title)

        # Save the plot
        output_file = get_results_full_path(
            os.path.join(OUTPUT_DIR, f"SP {y_label}.png")
        )
        save_plot(output_file)
        print(f"Saved plot '{title}' to {output_file}")

        # Clear the plot
        plt.clf()

    # Special plot: Scatterplot of nash social welfare vs conflicts
    df_plot = df_models.copy()
    df_plot["agent_model"] = df_plot["agent_model"].str.replace("\n", " ")
    df_plot = df_plot.rename(columns={"agent_model": "Agent Model"})
    initialize_plot_default()
    plt.rcParams["lines.marker"] = ""
    sns.regplot(
        data=df_plot,
        x="combat/game_conflicts_avg",
        y="benchmark/nash_social_welfare_global",
        scatter=False,
    )
    initialize_plot_default()
    plt.rcParams["lines.markersize"] = 12
    sns.scatterplot(
        data=df_plot,
        x="combat/game_conflicts_avg",
        y="benchmark/nash_social_welfare_global",
        hue="Agent Model",
        palette=MODEL_COMPARISON_COLORS,
    )
    plt.xlabel("Average Conflicts per Phase")
    plt.ylabel("Nash Social Welfare")
    plt.title("Nash Social Welfare vs Average Conflicts per Phase")
    title = "SP Nash Social Welfare vs Conflicts.png"
    output_file = get_results_full_path(os.path.join(OUTPUT_DIR, title))
    save_plot(output_file)
    print(f"Saved plot '{title}' to {output_file}")


if __name__ == "__main__":
    main()
