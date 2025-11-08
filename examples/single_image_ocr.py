#!/usr/bin/env python3
"""
Example: Process a single image programmatically.
Demonstrates basic usage of the Mario Kart Scoreboard OCR Pipeline.
"""
from src.orchestrator import OCRProcessor
from main import process_single_image


def main():
    """Process a single image with the OCR pipeline."""
    print("=" * 80)
    print("Single Image OCR Processing")
    print("=" * 80)

    # Initialize the OCR processor with config
    processor = OCRProcessor(
        config_path='src/configs/pipelines/default.json',
        debug=False  # Set to True for verbose logging
    )

    # Process a single image
    image_path = 'pngs/IMG_7995.png'

    try:
        # Use the helper function from main.py for consistent processing
        results = process_single_image(processor, image_path)

        print(f"\nProcessing complete!")
        print(f"  Extracted cells: {results['extracted_cells']}")
        print(f"  Valid predictions: {results['valid_predictions']}")
        print(f"  Invalid predictions: {results['invalid_predictions']}")
        print(f"  Failed cells: {results['failed_cells']}")
        print(f"\nOutput files:")
        print(f"  - Preprocessed image: output/preprocessed_images/IMG_7995_preprocessed.png")
        print(f"  - Annotated image: output/annotated_images/IMG_7995_annotated.jpg")
        print(f"  - Predictions CSV: {results['output_file']}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure pngs/IMG_7995.png exists")
    except Exception as e:
        print(f"Processing error: {e}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
