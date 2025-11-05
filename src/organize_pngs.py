#!/usr/bin/env python3
"""
Organize PNG files from heics subdirectories.
Renames .png files to their corresponding image names and moves them to pngs folder.
"""

import shutil
from pathlib import Path


def organize_pngs(heics_dir: str = 'heics', output_dir: str = 'pngs'):
    """
    Organize PNG files from heics subdirectories.

    Args:
        heics_dir: Directory containing HEIC files and subdirectories with PNGs
        output_dir: Directory to save renamed PNG files
    """
    heics_path = Path(heics_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(exist_ok=True)

    # Find all subdirectories
    subdirs = [d for d in heics_path.iterdir() if d.is_dir()]

    print(f"Found {len(subdirs)} subdirectories")

    moved = 0
    for subdir in subdirs:
        # Get the directory name (e.g., IMG_7995)
        dir_name = subdir.name

        # Find .png files in subdirectory
        png_files = list(subdir.glob('*.png'))

        if png_files:
            for png_file in png_files:
                # New name: use directory name as base (e.g., IMG_7995.png)
                new_name = f"{dir_name}.png"
                output_file = output_path / new_name

                # Copy file
                shutil.copy2(png_file, output_file)
                print(f"Copied: {png_file.relative_to(heics_path)} -> {output_file.relative_to('.')}")
                moved += 1

    print(f"\n✓ Moved {moved} PNG files to {output_dir}/")


if __name__ == '__main__':
    organize_pngs()
