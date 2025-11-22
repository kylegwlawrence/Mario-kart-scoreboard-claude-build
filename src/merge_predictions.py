"""
Prediction CSV file merging utility.

Provides functionality to combine multiple prediction CSV files from the OCR pipeline
into a single consolidated CSV file for analysis and reporting.
"""

import pandas as pd
from pathlib import Path
from src.custom_logger import get_custom_logger
import logging
import argparse


def merge_predictions_csvs(predictions_dir: str, merged_csv_path: str) -> None:
    """
    Merge multiple prediction CSV files into a single consolidated CSV.

    Reads all CSV files matching the pattern '*_predictions.csv' from the specified
    directory and combines them into a single output CSV file. Each row from the input
    files is preserved with a new index starting from 0.

    Args:
        predictions_dir: Path to directory containing prediction CSV files
        merged_csv_path: Path where the merged CSV file will be written

    Raises:
        FileNotFoundError: If predictions directory does not exist
        Exception: If file reading or writing operations fail
    """
    logger = get_custom_logger(name=__name__, level=logging.DEBUG, log_file='app.log')
    # get list of all csv files in the predictions directory
    predictions_dir = Path(predictions_dir)
    csv_files = list(predictions_dir.glob("*_predictions.csv"))
    # convert list of csv paths to list of dataframes and save
    try:
        df_list = [pd.read_csv(file) for file in csv_files]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(merged_csv_path, index=False)
    except Exception as e:
        logger.exception(e)
        raise
    logger.info(f"Written merged predictions to: {merged_csv_path}")

def main() -> None:
    """
    Command-line entry point for merging prediction CSV files.

    Parses command-line arguments for the predictions directory and output path,
    then calls merge_predictions_csvs to perform the merge operation.

    Command-line arguments:
        --predictions_dir: Directory containing prediction CSV files (optional)
        --merged_csv_path: Output path for the merged CSV file (optional)
    """
    parser = argparse.ArgumentParser(
        description="Merge multiple prediction CSV files into a single CSV file"
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        help="Directory containing prediction CSV files (default: output/predictions)"
    )
    parser.add_argument(
        "--merged_csv_path",
        type=str,
        help="Output path for the merged CSV file (default: analysis/all_predictions.csv)"
    )

    args = parser.parse_args()
    merge_predictions_csvs(args.predictions_dir, args.merged_csv_path)

if __name__ == "__main__":
    main()