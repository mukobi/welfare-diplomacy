"""Use the Weights & Biases API to download data from the same-policy runs."""

import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb


from chart_utils import (
    ALL_POWER_ABBREVIATIONS,
    get_results_full_path,
    MODEL_NAME_TO_DISPLAY_NAME,
)

INPUT_DIR = "../results/same_policy"
OUTPUT_DIR = "../results/same_policy_run_data"

ENTITIES_AND_PROJECTS_TO_TRY = [
    "gabrielmukobi/welfare-diplomacy-v3",
    "nlauffer/welfare-diplomacy-v2",
]

METRIC_KEYS = [
    f"score/{metric_type}/{power}"
    for metric_type in ["welfare", "centers", "units"]
    for power in ALL_POWER_ABBREVIATIONS
]


def main():
    """Main function."""

    # Load the data from each file into one big dataframe
    df_models = pd.concat(
        [
            pd.read_csv(get_results_full_path(os.path.join(INPUT_DIR, f)))
            for f in os.listdir(get_results_full_path(INPUT_DIR))
        ]
    )

    # Convert agent_names of super exploiter to Super Exploiter
    df_models.loc[
        df_models["super_exploiter_powers"].notnull(), "agent_model"
    ] = "Super Exploiter"

    # Print a table of runs per model
    print(df_models.groupby("agent_model").size())
    sum_runs = len(df_models)

    # Get the model names
    model_names = df_models["agent_model"].unique()

    # Initialize things
    api = wandb.Api()

    # Iterate over each model
    progress_bar = tqdm(total=sum_runs)
    for model_name in model_names:
        # Filter to the runs with this model
        df_model = df_models[df_models["agent_model"] == model_name]

        # Iterate over the run IDs
        all_row_data = []
        for run_id in df_model["ID"]:
            run = None
            for entity_project in ENTITIES_AND_PROJECTS_TO_TRY:
                try:
                    run = api.run(f"{entity_project}/{run_id}")
                    break
                except wandb.errors.CommError:
                    continue
            assert run is not None
            assert run.state == "finished"

            # Hack: We renamed the year_fractional metric, so we have to check both names
            progress_had_data = meta_had_data = False
            for i, row in run.history(
                keys=METRIC_KEYS + ["_progress/year_fractional"]
            ).iterrows():
                progress_had_data = True
                all_row_data.append(get_row_data(row, model_name))
            for i, row in run.history(
                keys=METRIC_KEYS + ["meta/year_fractional"]
            ).iterrows():
                meta_had_data = True
                all_row_data.append(get_row_data(row, model_name))
            assert (progress_had_data or meta_had_data) and not (
                progress_had_data and meta_had_data
            )
            progress_bar.update(1)

        # Save the data with the model name
        df = pd.DataFrame(all_row_data)
        if not os.path.exists(get_results_full_path(OUTPUT_DIR)):
            os.makedirs(get_results_full_path(OUTPUT_DIR))
        df.to_csv(
            get_results_full_path(
                os.path.join(OUTPUT_DIR, f"{model_name}_run_data.csv")
            ),
            index=False,
        )


def get_row_data(row, model_name: str):
    """Extract the desired data from a given row and return it as a dict."""
    output = {
        "agent_model": model_name,
        "year_integer": int(row["_progress/year_fractional"])
        if "_progress/year_fractional" in row
        else int(row["meta/year_fractional"]),
    }
    assert output["year_integer"] is not None
    assert output["year_integer"] != np.NaN
    assert output["year_integer"] >= 0.0
    output.update({metric: row[metric] for metric in METRIC_KEYS})
    return output


if __name__ == "__main__":
    main()
