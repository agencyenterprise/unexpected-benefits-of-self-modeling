import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import argparse  # Import argparse for command-line argument parsing


def tensorboard_logs_to_dataframe(log_dir, log_name, df):
    """Extracts all scalar data from tensorboard logs in a directory and returns
    a pandas DataFrame with the data.

    Args:
        log_dir (str): The directory containing the tensorboard logs.
        log_name (str): The name of the current subdirectory inside the log dir.
        df (pd.DataFrame): The DataFrame to append the data to.

    Returns:
        pd.DataFrame: The DataFrame with the extracted data.
    """
    run_dirs = [
        os.path.join(log_dir, d)
        for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d))
    ]
    for run_dir in run_dirs:
        ea = event_accumulator.EventAccumulator(
            run_dir,
            size_guidance={
                event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                event_accumulator.IMAGES: 4,
                event_accumulator.AUDIO: 4,
                event_accumulator.SCALARS: 0,
                event_accumulator.HISTOGRAMS: 1,
            },
        )
        ea.Reload()
        scalar_tags = ea.Tags()["scalars"]
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            tag_df = pd.DataFrame(
                {
                    "Log": log_name,
                    "Run": os.path.basename(run_dir),
                    "Tag": tag,
                    "Step": steps,
                    "Value": values,
                }
            )
            df = pd.concat([df, tag_df], ignore_index=True)
    return df


def extract_logs_to_dataframe(log_dir):
    """Extracts all scalar data from tensorboard logs in a directory and
    returns a pandas DataFrame with the data.

    Args:
        log_dir (str): The directory containing the tensorboard logs.

    Returns:
        pd.DataFrame: The DataFrame with the extracted data.
    """
    df = pd.DataFrame(columns=["Log", "Run", "Tag", "Step", "Value"])
    log_dirs = [
        os.path.join(log_dir, d)
        for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d))
    ]
    for log_name in log_dirs:
        df = tensorboard_logs_to_dataframe(log_name, log_name, df)
    return df


def plot_and_save(df, output_dir):
    """Plots the data from DataFrame and saves the plots to files.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be plotted.
        output_dir (str): Directory path to save the plots.
    """
    sns.set(style="whitegrid")

    # Assign unique colors to each combination of Log and Run
    color_palette = sns.color_palette(
        "husl", len(df["Log"].unique()) * len(df["Run"].unique())
    )
    color_dict = {}
    for i, (log, run) in enumerate(
        df[["Log", "Run"]].drop_duplicates().itertuples(index=False)
    ):
        color_dict[(log, run)] = color_palette[i]
    grouped_tag = df.groupby("Tag")
    for tag, tag_group_df in grouped_tag:
        fig, axes = plt.subplots(figsize=(10, 6))
        log_run_groups = tag_group_df.groupby(["Log", "Run"])
        for (log, run), log_run_group_df in log_run_groups:
            color = color_dict[(log, run)]
            sns.lineplot(
                x="Step",
                y="Value",
                data=log_run_group_df,
                label=f"{log} - {run}",
                ax=axes,
                color=color,
            )
        axes.set_title(tag)
        axes.set_xlabel("Step")
        axes.set_ylabel("Value")
        axes.legend()
        output_file = os.path.join(output_dir, f"{tag}.png")
        plt.savefig(output_file)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tensorboard logs.")
    parser.add_argument("--plot", action="store_true", help="Plot the data if given")
    args = parser.parse_args()

    log_dir = "artifacts/logs/"
    df = extract_logs_to_dataframe(log_dir)
    df.to_csv("artifacts/IMDB_run_logs.csv", index=False)

    if args.plot:
        plot_and_save(df, ".")
