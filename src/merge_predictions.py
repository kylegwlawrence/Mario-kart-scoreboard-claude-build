import pandas as pd
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger(
    name='MergePredictions',
    log_dir='.logging',
    debug=False,
    console_output=True
)

predictions_dir = Path("output/predictions")
csv_files = list(predictions_dir.glob("*_predictions.csv"))

df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

merged_csv_path = predictions_dir / "all_predictions.csv"
merged_df.to_csv(merged_csv_path, index=False)
logger.info(f"Written merged predictions to: {merged_csv_path}")
print(f"Merged {len(csv_files)} CSV files into all_predictions.csv")
