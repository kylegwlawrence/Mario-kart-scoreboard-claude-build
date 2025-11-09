import pandas as pd
from pathlib import Path
from utils import setup_logger
import argparse


def merge_predictions_csvs(predictions_dir:str, merged_csv_path:str) -> None:
    """
    Takes in a folder of predictions csv files, an output from the OCR pipeline, and merges them into one
    """
    logger = setup_logger(name='MergePredictions', log_dir='.logging', debug=False, console_output=True)

    # get list of all csv files in the predictions directory
    predictions_dir = Path(predictions_dir)
    csv_files = list(predictions_dir.glob("*_predictions.csv"))

    # convert list of csv paths to list of dataframes
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # union all csvs and save to merged_csv_path
    merged_df.to_csv(merged_csv_path, index=False)

    # log and print to console
    logger.info(f"Written merged predictions to: {merged_csv_path}")
    print(f"Merged {len(csv_files)} CSV files into all_predictions.csv")


def main():
    """Main entry point for command-line execution"""
    parser = argparse.ArgumentParser(
        description="Merge multiple prediction CSV files into a single CSV file"
    )
    parser.add_argument(
        "predictions_dir",
        type=str,
        help="Directory containing prediction CSV files (default: output/predictions)"
    )
    parser.add_argument(
        "merged_csv_path",
        type=str,
        help="Output path for the merged CSV file (default: analysis/all_predictions.csv)"
    )

    args = parser.parse_args()
    merge_predictions_csvs(args.predictions_dir, args.merged_csv_path)


if __name__ == "__main__":
    main()