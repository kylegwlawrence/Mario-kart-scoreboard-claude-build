#!/usr/bin/env python3
"""
Main entry point for Mario Kart Scoreboard OCR Pipeline.
Supports both CLI and programmatic usage.
"""

import sys
import argparse
from pathlib import Path

from src.orchestrator import OCRProcessor


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Mario Kart Scoreboard OCR Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image with default config
  python main.py --image pngs/IMG_7995.png

  # Process a single image with custom config
  python main.py --image pngs/IMG_7995.png --config src/configs/custom_config.json

  # Enable debug logging
  python main.py --image pngs/IMG_7995.png --debug

  # Process all images in a directory
  python main.py --image-dir pngs/ --config src/configs/default_config.json
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image to process'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Path to directory containing images to process'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/default_config.json',
        help='Path to configuration file (default: src/configs/default_config.json)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.image and not args.image_dir:
        parser.error('Either --image or --image-dir must be specified')

    if args.image and args.image_dir:
        parser.error('Cannot specify both --image and --image-dir')

    if not Path(args.config).exists():
        parser.error(f'Config file not found: {args.config}')

    # Initialize OCR processor
    try:
        processor = OCRProcessor(args.config, debug=args.debug)
    except Exception as e:
        print(f"Failed to initialize processor: {e}", file=sys.stderr)
        return 1

    # Process images
    try:
        # process one image
        if args.image:
            process_single_image(processor, args.image)
        # process all images in a directory
        else:
            process_image_directory(processor, args.image_dir)

    except Exception as e:
        processor.logger.error(f"Processing failed: {e}")
        return 1

    return 0


def process_single_image(processor: OCRProcessor, image_path: str) -> None:
    """
    Process a single image.

    Args:
        processor: OCRProcessor instance
        image_path: Path to image file

    Raises:
        FileNotFoundError: If image file not found
        Exception: If processing fails
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    processor.logger.info(f"\nProcessing single image: {image_path}")
    
    # process the image with the specified ocr processor and save to output
    processor.process_image(image_path)
    processor.logger.info(f"\nResults saved to {processor.output_paths['predictions']}/")


def process_image_directory(processor: OCRProcessor, image_dir: str) -> None:
    """
    Process all images in a directory.

    Args:
        processor: OCRProcessor instance
        image_dir: Path to directory containing images

    Raises:
        NotADirectoryError: If directory not found
        Exception: If processing fails (stops on first error)
    """
    image_dir_path = Path(image_dir)

    if not image_dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {image_dir}")

    # Find all PNG images
    image_files = sorted(image_dir_path.glob('*.png'))

    if not image_files:
        processor.logger.warning(f"No PNG images found in {image_dir}")
        return

    processor.logger.info(f"Found {len(image_files)} images to process")

    # loop over the pngs to apply the OCR engine
    for i, image_path in enumerate(image_files, 1):
        try:
            processor.logger.info(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            
            # process the image with the specified ocr processor and save to output
            results = processor.process_image(str(image_path))
            processor.logger.info(f"Success: {results['valid_predictions']}/{results['total_cells']} cells valid")

        except Exception as e:
            processor.logger.error(f"Processing failed for {image_path.name}: {e}")
            # Stop on first error as specified
            raise


if __name__ == '__main__':
    sys.exit(main())
