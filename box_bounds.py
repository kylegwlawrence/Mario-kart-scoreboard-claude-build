"""
Grid coordinates for 12 rows and 5 columns with variable column widths.
Each row has uniform height, but columns can have different widths.
"""

from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

# Grid configuration
NUM_ROWS = 12
NUM_COLUMNS = 5

# Define column boundaries as percentages of image width (0.0 to 1.0)
# Format: (left_percent, right_percent) for each column
COLUMN_BOUNDS = [
    (0.0, 0.065),    # Column 0: 0% to 15% of width
    (0.065, 0.16),   # Column 1: 15% to 35% of width
    (0.16, 0.74),   # Column 2: 35% to 65% of width
    (0.74, 0.9),   # Column 3: 65% to 85% of width
    (0.9, 1.0),    # Column 4: 85% to 100% of width
]

# Define row boundaries as percentages of image height (0.0 to 1.0)
# Format: (top_percent, bottom_percent) for each row
ROW_BOUNDS = [
    (0.0, 0.0833),      # Row 0
    (0.0833, 0.1667),   # Row 1
    (0.1667, 0.25),     # Row 2
    (0.25, 0.3333),     # Row 3
    (0.3333, 0.4167),   # Row 4
    (0.4167, 0.5),      # Row 5
    (0.5, 0.5833),      # Row 6
    (0.5833, 0.6667),   # Row 7
    (0.6667, 0.75),     # Row 8
    (0.75, 0.8333),     # Row 9
    (0.8333, 0.9167),   # Row 10
    (0.9167, 1.0),      # Row 11
]


def get_cell_bounds(row: int, col: int) -> Tuple[float, float, float, float]:
    """
    Get the bounding box coordinates for a specific cell as percentages.

    Args:
        row: Row index (0-11)
        col: Column index (0-4)

    Returns:
        Tuple of (left, top, right, bottom) as decimal percentages (0.0-1.0)
    """
    if not (0 <= row < NUM_ROWS and 0 <= col < NUM_COLUMNS):
        raise ValueError(f"Invalid cell position: row={row}, col={col}")

    col_left, col_right = COLUMN_BOUNDS[col]
    row_top, row_bottom = ROW_BOUNDS[row]

    return (col_left, row_top, col_right, row_bottom)


def get_all_cells() -> List[Tuple[int, Tuple[int, int, int, int]]]:
    """
    Get all cell coordinates for the entire grid.

    Returns:
        List of tuples: (cell_index, (left, top, right, bottom))
    """
    cells = []
    cell_idx = 0
    for row in range(NUM_ROWS):
        for col in range(NUM_COLUMNS):
            bounds = get_cell_bounds(row, col)
            cells.append((cell_idx, bounds))
            cell_idx += 1
    return cells


def draw_grid_bounds(image_path: str, output_path: str, line_color: str = "red", line_width: int = 2) -> None:
    """
    Draw column and row bounds on an image with integer index labels.

    Args:
        image_path: Path to the input image
        output_path: Path to save the annotated image
        line_color: Color of the grid lines (default: "red")
        line_width: Thickness of the grid lines in pixels (default: 2)
    """
    # Load the image
    img = Image.open(image_path)
    width, height = img.size

    # Create a copy to draw on
    annotated_img = img.copy()
    draw = ImageDraw.Draw(annotated_img)

    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Draw vertical lines for column boundaries with column index labels
    column_x_coords = set()
    for col_idx, (left_pct, right_pct) in enumerate(COLUMN_BOUNDS):
        left_x = int(left_pct * width)
        right_x = int(right_pct * width)

        # Add both boundaries to our set
        column_x_coords.add(left_x)
        column_x_coords.add(right_x)

    # Draw all vertical lines
    for x_coord in sorted(column_x_coords):
        draw.line([(x_coord, 0), (x_coord, height)], fill=line_color, width=line_width)

    # Add column index labels at the top
    for col_idx, (left_pct, right_pct) in enumerate(COLUMN_BOUNDS):
        left_x = int(left_pct * width)
        right_x = int(right_pct * width)
        center_x = (left_x + right_x) // 2
        label = str(col_idx)
        draw.text((center_x - 5, 5), label, fill=line_color, font=font)

    # Draw horizontal lines for row boundaries with row index labels
    row_y_coords = set()
    for row_idx, (top_pct, bottom_pct) in enumerate(ROW_BOUNDS):
        top_y = int(top_pct * height)
        bottom_y = int(bottom_pct * height)

        # Add both boundaries to our set
        row_y_coords.add(top_y)
        row_y_coords.add(bottom_y)

    # Draw all horizontal lines
    for y_coord in sorted(row_y_coords):
        draw.line([(0, y_coord), (width, y_coord)], fill=line_color, width=line_width)

    # Add row index labels on the left side
    for row_idx, (top_pct, bottom_pct) in enumerate(ROW_BOUNDS):
        top_y = int(top_pct * height)
        bottom_y = int(bottom_pct * height)
        center_y = (top_y + bottom_y) // 2
        label = str(row_idx)
        draw.text((5, center_y - 7), label, fill=line_color, font=font)

    # Save the annotated image
    annotated_img.save(output_path)
    print(f"Grid bounds drawn and saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print(f"Grid: {NUM_ROWS} rows � {NUM_COLUMNS} columns")
    print("All bounds defined as decimal percentages (0.0 to 1.0)")
    print()

    # Display column information
    print("Column bounds (as % of image width):")
    for i, (left, right) in enumerate(COLUMN_BOUNDS):
        print(f"  Column {i}: {left:.4f} to {right:.4f} (width: {(right - left)*100:.1f}%)")
    print()

    # Display row information
    print("Row bounds (as % of image height):")
    for i, (top, bottom) in enumerate(ROW_BOUNDS):
        print(f"  Row {i}: {top:.4f} to {bottom:.4f} (height: {(bottom - top)*100:.2f}%)")
    print()

    # Example: Get bounds for cell at row 0, column 0
    cell_bounds = get_cell_bounds(0, 0)
    print(f"Cell [0, 0] bounds: {cell_bounds}")

    # Example: Get bounds for cell at row 5, column 2
    cell_bounds = get_cell_bounds(5, 2)
    print(f"Cell [5, 2] bounds: {cell_bounds}")
    print()

    # Example: Draw grid bounds on an image
    print("Example: Drawing grid bounds on an image")
    print("Usage: draw_grid_bounds('path/to/image.jpg', 'path/to/output.jpg', line_color='red', line_width=2)")

    image_num = 8003
    # Uncomment below to test with an actual image file:
    draw_grid_bounds(f"pngs/IMG_{image_num}.png", f"output/annotated_images/IMG_{image_num}.jpg", line_color="red", line_width=2)
