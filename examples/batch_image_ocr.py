#!/usr/bin/env python3
"""
Example: Process multiple images from a directory.
Demonstrates batch processing with the Mario Kart Scoreboard OCR Pipeline.
"""
from pathlib import Path
from src.orchestrator import OCRProcessor
from main import process_image_directory


def main():
    """Process multiple images from a directory."""
    print("=" * 80)
    print("Batch Image OCR Processing")
    print("=" * 80)

    processor = OCRProcessor(
        config_path='src/configs/pipelines/default.json',
        debug=False
    )

    image_dir = Path('pngs')

    # Use the helper function from main.py for consistent batch processing
    # This function handles directory validation, image discovery, and error handling
    process_image_directory(processor, str(image_dir))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
