"""
Table detection and annotation module for Mario Kart scoreboard.
Detects table bounds and creates annotated images.
"""

import logging
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from src.config_manager import ConfigManager


class TableDetector:
    """Detects and extracts Mario Kart scoreboard table."""

    # Table constants
    TABLE_ROWS = 12
    TABLE_COLS = 3

    def __init__(
        self,
        config_manager: ConfigManager,
        enabled: bool = True,
        method: str = 'contour',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize table detector.

        Args:
            config_manager: Configuration manager instance
            enabled: Whether table detection is enabled
            method: Detection method ('contour' or 'manual')
            logger: Logger instance
        """
        self.config_manager = config_manager
        self.enabled = enabled
        self.method = method
        self.logger = logger

        if logger:
            logger.info(f"Initialized TableDetector (enabled={enabled}, method={method})")

    def find_table_bounds(
        self,
        image: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Find the bounding box of the table in the image.

        Args:
            image: Preprocessed image (preferably grayscale or edge-detected)

        Returns:
            Tuple of (x, y, width, height) representing table bounds

        Raises:
            ValueError: If table detection fails
        """
        if not self.enabled:
            # Return full image bounds
            h, w = image.shape[:2]
            return 0, 0, w, h

        try:
            if self.method == 'contour':
                return self._find_bounds_contour(image)
            else:
                raise ValueError(f"Unknown detection method: {self.method}")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Table detection failed: {e}. Using full image bounds.")
            # Fallback: return full image
            h, w = image.shape[:2]
            return 0, 0, w, h

    def _find_bounds_contour(
        self,
        image: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Find table bounds using contour detection.

        Args:
            image: Preprocessed image

        Returns:
            Tuple of (x, y, width, height)

        Raises:
            ValueError: If table is not found
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply edge detection if not already done
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in image")

        # Find large rectangular contours (likely table grid lines)
        # This handles cases where the table is made of multiple lines
        large_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Look for contours that are large, fairly rectangular (wide), and reasonably tall
            # Filter for lines/shapes that are likely parts of the table grid
            if area > 5000 and w > 200 and h > 20:
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio > 10:  # Horizontal lines have high aspect ratio
                    large_contours.append((x, y, w, h))

        if large_contours:
            # Find the bounding box that encompasses all large contours
            min_x = min(x for x, y, w, h in large_contours)
            min_y = min(y for x, y, w, h in large_contours)
            max_x = max(x + w for x, y, w, h in large_contours)
            max_y = max(y + h for x, y, w, h in large_contours)

            x, y = min_x, min_y
            w, h = max_x - min_x, max_y - min_y
        else:
            # Fallback: find the largest rectangular contour
            best_contour = None
            best_area = 0

            for contour in contours:
                # Approximate contour to a rectangle
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if it's roughly rectangular (4 sides)
                if len(approx) >= 4:
                    area = cv2.contourArea(contour)
                    if area > best_area:
                        best_area = area
                        best_contour = contour

            if best_contour is None:
                raise ValueError("No rectangular contour found")

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(best_contour)

        if self.logger:
            self.logger.debug(f"Found table bounds: x={x}, y={y}, w={w}, h={h}")

        return x, y, w, h

    def extract_cells(
        self,
        image: np.ndarray,
        table_bounds: Tuple[int, int, int, int],
        rows: int = TABLE_ROWS,
        cols: int = TABLE_COLS
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Extract individual cells from the table.

        Args:
            image: Original image (color or grayscale)
            table_bounds: (x, y, width, height) of table
            rows: Number of rows in table
            cols: Number of columns in table

        Returns:
            Dictionary with (row, col) as key and cell image as value

        Raises:
            ValueError: If table bounds are invalid
        """
        x, y, w, h = table_bounds

        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid table bounds: {table_bounds}")

        # Extract table region
        table_region = image[y:y+h, x:x+w]

        # Calculate cell dimensions
        cell_height = h // rows
        cell_width = w // cols

        cells = {}

        for row in range(rows):
            for col in range(cols):
                # Calculate cell coordinates
                cell_x = col * cell_width
                cell_y = row * cell_height
                cell_w = cell_width if col < cols - 1 else (w - cell_x)
                cell_h = cell_height if row < rows - 1 else (h - cell_y)

                # Extract cell
                cell = table_region[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w]
                cells[(row, col)] = cell

        if self.logger:
            self.logger.info(f"Extracted {len(cells)} cells from table")

        return cells

    def get_cell_coordinates(
        self,
        row: int,
        col: int,
        table_bounds: Tuple[int, int, int, int],
        rows: int = TABLE_ROWS,
        cols: int = TABLE_COLS
    ) -> Tuple[int, int, int, int]:
        """
        Get pixel coordinates of a cell in the original image.

        Args:
            row: Row index
            col: Column index
            table_bounds: (x, y, width, height) of table
            rows: Number of rows in table
            cols: Number of columns in table

        Returns:
            Tuple of (x, y, width, height) in original image coordinates
        """
        table_x, table_y, table_w, table_h = table_bounds

        cell_height = table_h // rows
        cell_width = table_w // cols

        cell_x = table_x + col * cell_width
        cell_y = table_y + row * cell_height
        cell_w = cell_width if col < cols - 1 else (table_w - col * cell_width)
        cell_h = cell_height if row < rows - 1 else (table_h - row * cell_height)

        return cell_x, cell_y, cell_w, cell_h

    def annotate_image(
        self,
        image: np.ndarray,
        predictions: Dict[Tuple[int, int], Tuple[str, float, bool, int, Dict, List[str]]],
        table_bounds: Tuple[int, int, int, int],
        rows: int = TABLE_ROWS,
        cols: int = TABLE_COLS,
        predicted_text_size: float = 3,
        predicted_text_thickness: int = 8,
        conf_text_size: float = 1,
        conf_text_thickness: int = 3,
        font_color: Tuple[int, int, int] = (0, 0, 150)
    ) -> np.ndarray:
        """
        Create annotated image with gridlines and predicted text.

        Args:
            image: Original image to annotate
            predictions: Dictionary with (row, col) as key and (text, confidence, passes_validation, retry_attempt, chain_config, cell_image_paths) as value
            table_bounds: (x, y, width, height) of table
            rows: Number of rows in table
            cols: Number of columns in table
            predicted_text_size: Font scale for predicted text (default: 3)
            predicted_text_thickness: Font thickness for predicted text (default: 5)
            conf_text_size: Font scale for confidence score text (default: 1.5)
            conf_text_thickness: Font thickness for confidence score text (default: 2)
            font_color: RGB color tuple for text (default: (0, 0, 255) - red)

        Returns:
            Annotated image

        Raises:
            ValueError: If image or predictions are invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image for annotation")

        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            annotated = image.copy()

        # Lighten the image by blending with white for better text visibility
        white_overlay = np.ones_like(annotated) * 255
        annotated = cv2.addWeighted(annotated, 0.7, white_overlay, 0.3, 0)

        # Get image dimensions for percentage-to-pixel conversion
        img_height, img_width = annotated.shape[:2]

        # Draw gridlines using config bounds definitions
        grid_color = (0, 0, 255)  # Red
        grid_thickness = 2

        column_bounds = self.config_manager.get_column_bounds()
        row_bounds = self.config_manager.get_row_bounds()
        num_rows = self.config_manager.get_num_rows()
        num_columns = self.config_manager.get_num_columns()

        # Draw vertical lines based on column bounds
        for col in range(len(column_bounds)):
            left_pct, _ = column_bounds[col]
            x = int(left_pct * img_width)
            cv2.line(annotated, (x, 0), (x, img_height), grid_color, grid_thickness)

        # Draw right edge for last column
        _, right_pct = column_bounds[-1]
        x = int(right_pct * img_width)
        cv2.line(annotated, (x, 0), (x, img_height), grid_color, grid_thickness)

        # Draw horizontal lines based on row bounds
        for row in range(len(row_bounds)):
            top_pct, _ = row_bounds[row]
            y = int(top_pct * img_height)
            cv2.line(annotated, (0, y), (img_width, y), grid_color, grid_thickness)

        # Draw bottom edge for last row
        _, bottom_pct = row_bounds[-1]
        y = int(bottom_pct * img_height)
        cv2.line(annotated, (0, y), (img_width, y), grid_color, grid_thickness)

        # Add predicted text with confidence scores
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = predicted_text_size
        conf_font_scale = conf_text_size
        font_thickness = predicted_text_thickness
        conf_font_thickness = conf_text_thickness

        for (row, col), (text, confidence, _, _, _, _) in predictions.items():
            if 0 <= row < num_rows and 0 <= col < num_columns:
                # Use config_manager to get cell coordinates based on actual column indices
                left_pct, top_pct, right_pct, bottom_pct = self.config_manager.get_cell_bounds(row, col)
                cell_x = int(left_pct * img_width)
                cell_y = int(top_pct * img_height)
                cell_w = int((right_pct - left_pct) * img_width)
                cell_h = int((bottom_pct - top_pct) * img_height)

                # Add text label centered in cell
                text_label = f"{text}"
                text_size = cv2.getTextSize(text_label, font, font_scale, font_thickness)[0]

                # Add confidence score below text
                conf_label = f"({confidence:.2f})"
                conf_size = cv2.getTextSize(conf_label, font, conf_font_scale, conf_font_thickness)[0]

                # Calculate total height needed for both texts
                line_spacing = 5
                total_text_height = text_size[1] + conf_size[1] + line_spacing

                # Center both texts vertically within the cell
                text_y = cell_y + (cell_h - total_text_height) // 2 + text_size[1]
                conf_y = text_y + line_spacing + conf_size[1]

                # Center text horizontally
                text_x = cell_x + (cell_w - text_size[0]) // 2

                cv2.putText(
                    annotated,
                    text_label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    font_color,
                    font_thickness
                )

                # Center confidence score horizontally
                conf_x = cell_x + (cell_w - conf_size[0]) // 2

                cv2.putText(
                    annotated,
                    conf_label,
                    (conf_x, conf_y),
                    font,
                    conf_font_scale,
                    font_color,
                    conf_font_thickness
                )

        if self.logger:
            self.logger.info(f"Created annotated image with {len(predictions)} predictions")

        return annotated

