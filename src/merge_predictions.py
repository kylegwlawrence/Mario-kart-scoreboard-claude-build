import pandas as pd
from pathlib import Path
from src.custom_logger import get_custom_logger
import logging
import argparse

def merge_predictions_csvs(predictions_dir:str, merged_csv_path:str) -> None:
    """
    Takes in a folder of predictions csv files, an output from the OCR pipeline, and merges them into one
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

def main():
    """Main entry point for command-line execution"""
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