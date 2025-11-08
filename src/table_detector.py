"""
Table detection and annotation module for Mario Kart scoreboard.
Detects table bounds and creates annotated images.
"""

import logging
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
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

    def annotate_image(
        self,
        image: np.ndarray,
        predictions: Dict[Tuple[int, int], Tuple[str, float, bool, int, Dict, List[str]]],
        table_bounds: Tuple[int, int, int, int],
        predicted_text_size: float = 3,
        predicted_text_thickness: int = 8,
        conf_text_size: float = 1,
        conf_text_thickness: int = 3,
        font_color: Tuple[int, int, int] = (0, 0, 190)
    ) -> np.ndarray:
        """
        Create annotated image with gridlines and predicted text.

        Args:
            image: Original image to annotate
            predictions: Dictionary with (row, col) as key and (text, confidence, passes_validation, retry_attempt, chain_config, cell_image_paths) as value
            table_bounds: Table bounds tuple (x, y, width, height)
            predicted_text_size: Font scale for predicted text (default: 3)
            predicted_text_thickness: Font thickness for predicted text (default: 8)
            conf_text_size: Font scale for confidence score text (default: 1)
            conf_text_thickness: Font thickness for confidence score text (default: 3)
            font_color: RGB color tuple for text (default: (0, 0, 190) - red)

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

