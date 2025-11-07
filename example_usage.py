#!/usr/bin/env python3
"""
Example usage of the Mario Kart Scoreboard OCR Pipeline.
Demonstrates programmatic usage (non-CLI).
"""

from pathlib import Path
from src.orchestrator import OCRProcessor


def example_single_image_processing():
    """Example: Process a single image programmatically."""
    print("=" * 80)
    print("Example 1: Single Image Processing")
    print("=" * 80)

    # Initialize the OCR processor with config
    processor = OCRProcessor(
        config_path='src/configs/pipelines/default.json',
        debug=False  # Set to True for verbose logging
    )

    # Process a single image
    image_path = 'pngs/IMG_7995.png'

    try:
        results = processor.process_image(image_path)

        print(f"\nProcessing complete!")
        print(f"  Total cells: {results['total_cells']}")
        print(f"  Valid predictions: {results['valid_predictions']}")
        print(f"  Failed cells: {results['failed_cells']}")
        print(f"  Process time: {results['process_start_time']} to {results['process_end_time']}")
        print(f"\nOutput files:")
        print(f"  - Preprocessed image: output/preprocessed_images/IMG_7995_preprocessed.png")
        print(f"  - Annotated image: output/annotated_images/IMG_7995_annotated.jpg")
        print(f"  - Predictions CSV: output/predictions/IMG_7995_predictions.csv")
        print(f"  - Metadata JSON: output/predictions/IMG_7995_metadata.json")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure pngs/IMG_7995.png exists")
    except Exception as e:
        print(f"Processing error: {e}")


def example_batch_processing():
    """Example: Process multiple images from a directory."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Processing")
    print("=" * 80)

    processor = OCRProcessor(
        config_path='src/configs/pipelines/default.json',
        debug=False
    )

    image_dir = Path('pngs')

    if not image_dir.is_dir():
        print(f"Error: Directory {image_dir} not found")
        return

    # Get all PNG images
    image_files = sorted(image_dir.glob('*.png'))

    if not image_files:
        print(f"No PNG images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images to process\n")

    # Process each image
    successful = 0
    failed = 0

    for i, image_path in enumerate(image_files[:3], 1):  # Process first 3 for example
        try:
            print(f"[{i}] Processing {image_path.name}...", end=" ")
            results = processor.process_image(str(image_path))
            print(f"OK ({results['valid_predictions']}/{results['total_cells']} cells)")
            successful += 1

        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
            # As specified, stop on first error
            print("\nStopping on first error as configured")
            break

    print(f"\nResults: {successful} successful, {failed} failed")


def example_with_custom_config():
    """Example: Using a custom configuration."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Configuration")
    print("=" * 80)

    # You can create a custom config file with different:
    # - Preprocessing chains
    # - OCR engine parameters
    # - Output paths
    # - Retry strategies

    print("""
To use a custom configuration:

1. Create a new JSON file in the src/configs/pipelines/ directory
   e.g., src/configs/pipelines/my_custom_pipeline.json

2. Customize settings like:
   - Different preprocessing chain
   - Alternative OCR engine
   - Custom output paths
   - Different validation thresholds

3. Initialize processor with custom config:
   processor = OCRProcessor('src/configs/pipelines/my_custom_pipeline.json')

Note: Grid bounds and character names are centralized in:
   - src/configs/grid.json (grid structure - never changes)
   - src/configs/ocr_engines.json (OCR engine configs and character list)

Example custom pipeline config structure:
{
  "image_source": "./pngs",
  "output_paths": {
    "preprocessed": "output/preprocessed_images",
    "annotated": "output/annotated_images",
    "predictions": "output/predictions",
    "logs": ".logging"
  },
  "primary_engine": "tesseract",
  "retry_attempts": 5,
  "preprocessing_chains": [
    {
      "retry_attempt": 0,
      "methods": [
        {"method": "grayscale", "parameters": null},
        {"method": "bilateral_filter", "parameters": {"d": 9, "sigmaColor": 75, "sigmaSpace": 75}}
      ]
    }
  ]
}
    """)


def example_accessing_results():
    """Example: Accessing and working with results."""
    print("\n" + "=" * 80)
    print("Example 4: Accessing Results")
    print("=" * 80)

    print("""
After processing, the following files are generated:

1. Predictions CSV (output/predictions/{image}_predictions.csv)
   Columns:
   - row_id: Row in the table (0-11)
   - column_id: Column in the table (0-2)
   - predicted_text: Extracted and validated text
   - confidence: OCR confidence score (0-1)
   - text_coordinates: Pixel coordinates in original image
   - original_filepath: Path to original image
   - preprocessed_filepath: Path to preprocessed image
   - process_start_timestamp: When processing started
   - preprocessing_methods: Preprocessing steps applied

2. Metadata JSON (output/predictions/{image}_metadata.json)
   Contains:
   - Image paths and processing times
   - Configuration used (preprocessing, OCR, retry settings)
   - Summary of results (valid/failed cells)
   - Details of failed cells

3. Preprocessed PNG (output/preprocessed_images/{image}_preprocessed.png)
   The image after all preprocessing transformations

4. Annotated JPG (output/annotated_images/{image}_annotated.jpg)
   Original image with:
   - Green grid lines showing table structure
   - Red text showing OCR predictions
   - Confidence scores below each prediction

Example: Reading results programmatically
    import csv
    import json

    # Read predictions
    with open('output/predictions/IMG_7995_predictions.csv') as f:
        reader = csv.DictReader(f)
        predictions = list(reader)

    # Read metadata
    with open('output/predictions/IMG_7995_metadata.json') as f:
        metadata = json.load(f)
    """)


if __name__ == '__main__':
    # Run examples
    try:
        example_single_image_processing()
        # example_batch_processing()  # Uncomment to run
        example_with_custom_config()
        example_accessing_results()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
