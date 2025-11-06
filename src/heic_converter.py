#!/usr/bin/env python3
"""
HEIC to PNG image converter.
Converts Apple's HEIC/HEIF image format to PNG.
"""

import logging
from pathlib import Path
from typing import Optional
from PIL import Image
import pillow_heif


def convert_heic_to_png(
    input_path: str,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Convert HEIC image to PNG format.

    Args:
        input_path: Path to HEIC file
        output_path: Path to save PNG file. If None, uses same name with .png extension
        logger: Logger instance for logging

    Returns:
        Path to the converted PNG file

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file is not HEIC format
        IOError: If conversion or saving fails
    """
    # Validate input file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_file.suffix.lower() not in ['.heic', '.heif']:
        raise ValueError(f"Input file must be HEIC/HEIF format (case-insensitive), got: {input_file.suffix}")

    # Determine output path
    if output_path is None:
        output_path = str(input_file.with_suffix('.png'))

    output_file = Path(output_path)

    # Register HEIF/HEIC support
    pillow_heif.register_heif_opener()

    try:
        # Open HEIC image
        if logger:
            logger.info(f"Converting HEIC to PNG: {input_path}")

        image = Image.open(input_file)

        # Convert to RGB if necessary (HEIC might have alpha channel)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Save as PNG
        output_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_file, 'PNG')

        if logger:
            logger.info(f"Successfully converted to PNG: {output_path}")

        return str(output_file)

    except Exception as e:
        raise IOError(f"Failed to convert HEIC to PNG: {e}")


def batch_convert_heic_to_png(
    input_dir: str,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> list:
    """
    Convert all HEIC files in a directory to PNG.

    Args:
        input_dir: Directory containing HEIC files
        output_dir: Directory to save PNG files. If None, uses same directory
        logger: Logger instance for logging

    Returns:
        List of paths to converted PNG files

    Raises:
        NotADirectoryError: If input directory doesn't exist
        IOError: If any conversion fails
    """
    input_directory = Path(input_dir)
    if not input_directory.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    if output_dir is None:
        output_directory = input_directory
    else:
        output_directory = Path(output_dir)

    # Find all HEIC files (case-insensitive)
    heic_files = list(input_directory.glob('*.heic')) + list(input_directory.glob('*.heif')) + \
                 list(input_directory.glob('*.HEIC')) + list(input_directory.glob('*.HEIF'))

    if not heic_files:
        if logger:
            logger.warning(f"No HEIC/HEIF files found in {input_dir}")
        return []

    converted_files = []

    if logger:
        logger.info(f"Found {len(heic_files)} HEIC files to convert")

    for heic_file in heic_files:
        try:
            output_path = output_directory / heic_file.stem / '.png'
            png_path = convert_heic_to_png(str(heic_file), str(output_path), logger)
            converted_files.append(png_path)

        except Exception as e:
            if logger:
                logger.error(f"Failed to convert {heic_file}: {e}")
            raise

    if logger:
        logger.info(f"Successfully converted {len(converted_files)} files")

    return converted_files


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Convert HEIC images to PNG format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python heic_converter.py --input photo.heic

  # Convert single file to specific output
  python heic_converter.py --input photo.heic --output photo.png

  # Convert all HEIC files in directory
  python heic_converter.py --input-dir ./images --output-dir ./converted
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Path to input HEIC file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output PNG file'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Directory containing HEIC files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save PNG files'
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        if args.input:
            # Convert single file
            png_path = convert_heic_to_png(args.input, args.output, logger)
            print(f"Converted: {png_path}")

        elif args.input_dir:
            # Convert directory
            converted = batch_convert_heic_to_png(args.input_dir, args.output_dir, logger)
            print(f"Converted {len(converted)} files")
            for png_path in converted:
                print(f"  - {png_path}")

        else:
            parser.error('Either --input or --input-dir must be specified')

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
