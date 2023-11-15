"""
Charts for Same Policy experiments.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from chart_utils import (
    ALL_POWER_ABBREVIATIONS,
    COLOR_ALT_1,
    COLOR_ALT_2,
    MODEL_NAME_TO_COLOR,
    MODEL_NAME_TO_DISPLAY_NAME,
    MODEL_ORDER,
    initialize_plot_bar,
    initialize_plot_default,
    save_plot,
    get_results_full_path,
    stderr,
)

INPUT_FILES_MODELS = [
    "../results/same_policy/SP Claude Both.csv",
    "../results/same_policy/SP GPT-3.5-Turbo-16K-0613.csv",
    "../results/same_policy/SP GPT-4-0613.csv",
    "../results/same_policy/SP GPT-4-Base.csv",
    "../results/same_policy/SP SuperExploiter (GPT-4).csv",
    "../results/same_policy/SP Llama2-70b-chat.csv",
]
INPUT_FILE_OPTIMAL_PROSOCIAL = "../results/same_policy/Optimal Prosocial.csv"
INPUT_FILE_RANDOM = "../results/same_policy/SP Random.csv"

OUTPUT_DIR = "same_policy"
NON_ROOT_WELFARE = False


def main() -> None:
    """Main function."""

    # Load the data from each file into one big dataframe
    df_models = pd.concat(
        [pd.read_csv(get_results_full_path(f)) for f in INPUT_FILES_MODELS]
    )

    # Preprocess a 2D table that has the WP for each [language model, power] combo for each run
    # model is in agent_model, and power welfares are each score/welfare/{power abbreviation}
    wp_column_names = [f"score/welfare/{power}" for power in ALL_POWER_ABBREVIATIONS]
    columns_of_interest = [
        "agent_model",
        "_progress/year_fractional",
        "super_exploiter_powers",
    ] + wp_column_names
    df_model_power_wp = df_models[columns_of_interest].copy()
    df_model_power_wp.loc[
        df_model_power_wp["super_exploiter_powers"].notnull(), "agent_model"
    ] = "Super Exploiter"
    df_model_power_wp = df_model_power_wp.drop(columns=["super_exploiter_powers"])
    # Calculate integer years, first converting NaN values to the value in the first row
    df_model_power_wp = df_model_power_wp.fillna(
        df_model_power_wp["_progress/year_fractional"].iloc[0]
    )
    df_model_power_wp["_progress/year_integer"] = df_model_power_wp[
        "_progress/year_fractional"
    ].apply(int)
    # Divide welfares by int years passed, like years_passed = int(row["_progress/year_fractional"])
    df_model_power_wp[wp_column_names] = df_model_power_wp[wp_column_names].div(
        df_model_power_wp["_progress/year_integer"], axis=0
    )
    # Rename to yearly_welfare
    df_model_power_wp = df_model_power_wp.rename(
        columns={
            wp: f"yearly_welfare/{power}"
            for wp, power in zip(wp_column_names, ALL_POWER_ABBREVIATIONS)
        }
    )
    # df_model_power_wp = df_model_power_wp.groupby(["agent_model"]).mean().reset_index()
    df_model_power_wp.to_csv(
        get_results_full_path(os.path.join(OUTPUT_DIR, "SP Model Power WP.csv")),
    )

    # Load other data
    df_optimal_prosocial = pd.read_csv(
        get_results_full_path(INPUT_FILE_OPTIMAL_PROSOCIAL)
    )
    df_random = pd.read_csv(get_results_full_path(INPUT_FILE_RANDOM))
    df_random["agent_model"] = "Random Policy"

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

    # Print table of mean and stdderr for each benchmark metric (basic competence components and nash welfare)
    print("\nTable of mean and stdderr for each benchmark metric:")
    for model_name in MODEL_ORDER:
        if any([name in model_name for name in ["Exploiter", "Optimal", "Random"]]):
            continue
        # Filter to the runs with this model
        df_model = df_models[df_models["agent_model"] == model_name]
        # Print the mean and stdderr for each metric
        # Fraction of centers owned is conquest/centers_owned_ratio
        fraction_centers_owned_values = df_model["conquest/centers_owned_ratio"]
        # Fraction of valid orders is orders/game_valid_ratio
        fraction_valid_orders_values = df_model["orders/game_valid_ratio"]
        # valid JSON is model/game_completion_non_error_ratio
        fraction_valid_json_values = df_model["model/game_completion_non_error_ratio"]
        # Competence score is benchmark/competence_score
        competence_score_values = df_model["benchmark/competence_score"]
        # Nash welfare is benchmark/nash_social_welfare_global
        nash_welfare_values = df_model["benchmark/nash_social_welfare_global"]

        # Print the mean and stdderr for each metric as latex
        # correct JSON, then valid orders, then fraction SCs, then competence, the welfare
        model_name_no_newline = model_name.replace("\n", " ")
        print(
            f"\t{model_name_no_newline} & {np.mean(fraction_valid_json_values):.3f} $\\pm$ {stderr(fraction_valid_json_values):.3f} & {np.mean(fraction_valid_orders_values):.3f} $\\pm$ {stderr(fraction_valid_orders_values):.3f} & {np.mean(fraction_centers_owned_values):.3f} $\\pm$ {stderr(fraction_centers_owned_values):.3f} & {np.mean(competence_score_values):.3f} $\\pm$ {stderr(competence_score_values):.3f} & {np.mean(nash_welfare_values):.3f} $\\pm$ {stderr(nash_welfare_values):.3f} \\\\[2.5pt]"
        )
        print("\t\hline")

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
            "benchmark/nash_social_welfare_global",
            "Root Nash Welfare",
            1,
            True,
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
            "Average Conflicts per Turn",
            -1,
            False,
            True,
            (None, None),
            "best",
        ),
    ]:
        # Initialize
        initialize_plot_bar()

        # Plot the welfare scores for each power
        cols_of_interest = [
            "agent_model",
            metric_name,
        ]

        plot_df = df_models[cols_of_interest].copy()

        # Exponentiate nash welfare scores
        if NON_ROOT_WELFARE and "welfare" in metric_name:
            plot_df[metric_name] = plot_df[metric_name].pow(7)
            df_random[metric_name] = df_random[metric_name].pow(7)
            df_optimal_prosocial[metric_name] = df_optimal_prosocial[metric_name].pow(7)
            plt.yscale("log")

        # update the column names
        x_label = "Agent Model"
        plot_df.columns = [x_label, y_label]

        # Create the plot
        model_order = MODEL_ORDER
        if "Optimal Prosocial" in model_order:
            model_order.remove("Optimal Prosocial")
        if "Random Policy" in model_order:
            model_order.remove("Random Policy")
        plot = sns.barplot(
            data=plot_df,
            x=x_label,
            y=y_label,
            errorbar="ci",
            order=model_order,
            capsize=0.2,
            hue=x_label,
            palette=MODEL_NAME_TO_COLOR,
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
        plt.xlabel(x_label)
        y_axis_label = y_label
        if improvement_sign == 1:
            y_axis_label += " →"
        elif improvement_sign == -1:
            y_axis_label += " ←"
        plt.ylabel(y_axis_label)
        title = f"{y_label} by Agent Model (Self-Play)"
        plt.title(title)

        # Set y bounds
        if y_bounds[0] is not None:
            plt.ylim(bottom=y_bounds[0])
        if y_bounds[1] is not None:
            plt.ylim(top=y_bounds[1])

        # Save the plot
        output_file = get_results_full_path(
            os.path.join(OUTPUT_DIR, f"SP {y_label}.png")
        )
        save_plot(output_file)
        print(f"Saved plot '{title}' to {output_file}")

        # Clear the plot
        plt.clf()

    # Special plot: Scatterplot of roow nash welfare vs other things
    for x_axis, x_label in [
        ("combat/game_conflicts_avg", "Conflicts"),
        ("conquest/game_centers_lost_avg", "SCs Stolen"),
    ]:
        df_plot = pd.concat([df_models, df_random]).copy()
        df_plot["agent_model"] = df_plot["agent_model"].str.replace("\n", " ")
        grouping = "Agent Model"
        df_plot = df_plot.rename(columns={"agent_model": grouping})
        initialize_plot_default()
        plt.rcParams["lines.marker"] = ""
        sns.regplot(
            data=df_plot,
            x=x_axis,
            y="benchmark/nash_social_welfare_global",
            scatter=False,
            color=COLOR_ALT_1,
        )
        initialize_plot_default()
        plt.rcParams["lines.markersize"] = 14
        sns.scatterplot(
            data=df_plot,
            x=x_axis,
            y="benchmark/nash_social_welfare_global",
            hue=grouping,
            style=grouping,
            palette=MODEL_NAME_TO_COLOR,
        )
        plt.xlabel(f"Average {x_label} per Turn ↓")
        plt.ylabel("Root Nash Welfare →")
        plt.title(f"Root Nash Welfare vs Average {x_label} per Turn (Self-Play)")
        # Legend in 2 columns
        plt.legend(
            borderaxespad=0.0,
            ncol=2,
            handletextpad=0.1,
            columnspacing=0.5,
        )
        title = f"SP Root Nash Welfare vs {x_label}.png"
        output_file = get_results_full_path(os.path.join(OUTPUT_DIR, title))
        save_plot(output_file)
        print(f"Saved plot '{title}' to {output_file}")

        plt.clf()


if __name__ == "__main__":
    main()
