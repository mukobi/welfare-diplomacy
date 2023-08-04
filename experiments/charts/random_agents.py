"""
Plot elo scores as for each model type.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from chart_utils import initialize_plot_bar, save_plot, get_results_full_path

INPUT_FILE = "../results/sweeps/Random Agent Random Seed.csv"
OUTPUT_DIR = "./environment/random_agents"


def main() -> None:
    """Main function."""

    # Initialize
    initialize_plot_bar()

    # Load the data
    df = pd.read_csv(get_results_full_path(INPUT_FILE))
    print(f"Loaded {INPUT_FILE} with shape {df.shape}")

    # Plot a bunch of different bar graphs for different metrics
    for metric_name, y_label in [
        ("welfare", "Welfare Points"),
        ("centers", "Supply Centers"),
        ("units", "Unit Counts"),
    ]:
        # Plot the welfare scores for each power
        cols_of_interest = [
            f"score/{metric_name}/AUS",
            f"score/{metric_name}/ENG",
            f"score/{metric_name}/FRA",
            f"score/{metric_name}/GER",
            f"score/{metric_name}/ITA",
            f"score/{metric_name}/RUS",
            f"score/{metric_name}/TUR",
        ]

        welfare_df = df[cols_of_interest].copy()

        # convert to long format
        welfare_df = welfare_df.melt()

        # update the column names
        welfare_df.columns = ["Power", y_label]

        # remove the 'score/welfare/' from the Power names
        welfare_df["Power"] = welfare_df["Power"].str.replace(
            f"score/{metric_name}/", ""
        )

        # Create the plot
        plot = sns.barplot(
            data=welfare_df,
            x="Power",
            y=y_label,
            capsize=0.1,
            errorbar="ci",
        )

        # Print the mean and error of each bar
        power_names = welfare_df["Power"].unique()
        for bar_label, bar, line in zip(power_names, plot.patches, plot.lines[::3]):
            error = (line.get_ydata()[1] - line.get_ydata()[0]) / 2
            print(f"{y_label} ({bar_label}): {bar.get_height():.2f} ± {error:.2f}")
        print()

        # Set labels and title
        plt.xlabel("Power")
        plt.ylabel(y_label)
        title = f"{y_label} by Power - Random Agent Sweep ($N=64$)"
        plt.title(title)

        # Save the plot
        output_file = OUTPUT_DIR + f"/{metric_name}.png"
        save_plot(output_file)
        print(f"Saved plot '{title}' to {output_file}")

        # Clear the plot
        plt.clf()

    # Also plot the min, median, mean, and max aggregated welfare scores.
    cols_of_interest = [
        "welfare/min",
        "welfare/median",
        "welfare/mean",
        "welfare/max",
    ]

    welfare_df = df[cols_of_interest].copy()

    # convert to long format
    welfare_df = welfare_df.melt()

    # update the column names
    bar_label = "Aggregation Type"
    y_label = "Welfare Points →"
    welfare_df.columns = [bar_label, y_label]

    # Rename the aggregation types
    welfare_df[bar_label] = (
        welfare_df[bar_label].str.replace("welfare/", "").str.title()
    )

    # Create the plot
    plot = sns.barplot(
        data=welfare_df,
        x=bar_label,
        y=y_label,
        capsize=0.1,
        errorbar="ci",
    )

    # Print the mean and error of each bar
    bar_labels = welfare_df[bar_label].unique()
    for bar_label, bar, line in zip(bar_labels, plot.patches, plot.lines[::3]):
        error = (line.get_ydata()[1] - line.get_ydata()[0]) / 2
        print(f"{y_label} ({bar_label}): {bar.get_height():.2f} ± {error:.2f}")
    print()

    # Set labels and title
    plt.xlabel(bar_label)
    plt.ylabel(y_label)
    title = "Aggregated Welfare Points - Random Agent Sweep ($N=64$)"
    plt.title(title)

    # Save the plot
    output_file = OUTPUT_DIR + "/aggregated_welfare.png"
    save_plot(output_file)
    print(f"Saved plot '{title}' to {output_file}")

    # Clear the plot
    plt.clf()


if __name__ == "__main__":
    main()
