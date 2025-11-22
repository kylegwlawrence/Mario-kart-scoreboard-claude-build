"""Image annotation module for Mario Kart scoreboard analysis.

This module provides functionality to annotate scoreboard images with grid lines,
predicted text results, and confidence scores. It uses configuration-based bounds
to draw grid overlays matching the table structure and displays OCR predictions
with their confidence levels.

Classes:
    ImageAnnotator: Annotates images with grid lines and OCR predictions.
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from src.config_manager import ConfigManager


class ImageAnnotator:
    """Annotates Mario Kart scoreboard images with grid lines and OCR predictions.

    Overlays grid lines based on configured table bounds and displays OCR
    predictions with confidence scores at their corresponding cell locations.
    The image is lightened for improved text visibility.

    Attributes:
        config_manager (ConfigManager): Manages table bounds and grid configuration.
        logger (Optional[logging.Logger]): Logger for tracking operations.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the ImageAnnotator.

        Args:
            config_manager (ConfigManager): Configuration manager providing table
                bounds, row/column definitions, and cell coordinate information.
            logger (Optional[logging.Logger]): Logger instance for logging operations.
                Defaults to None.
        """
        self.config_manager = config_manager
        self.logger = logger

        if logger:
            logger.info(f"Initialized ImageAnnotator")

    def annotate_image(
        self,
        image: np.ndarray,
        predictions: Dict[Tuple[int, int], Tuple[list, list]],
        predicted_text_size: float = 3,
        predicted_text_thickness: int = 8,
        conf_text_size: float = 1,
        conf_text_thickness: int = 3,
        font_color: Tuple[int, int, int] = (0, 0, 255)
    ) -> np.ndarray:
        """Create annotated image with grid lines and OCR predictions.

        Overlays a grid based on configured table bounds and annotates each cell
        with the best OCR prediction and its confidence score. The image is
        lightened by blending with white to improve text visibility.

        Args:
            image (np.ndarray): Original scoreboard image to annotate.
            predictions (Dict[Tuple[int, int], Tuple[list, list]]): Dictionary mapping
                (row, col) cell coordinates to tuples of (all_attempts, cell_image_paths),
                where all_attempts is a list of dicts containing 'text' and 'confidence'.
            predicted_text_size (float): Font scale for OCR text labels. Defaults to 3.
            predicted_text_thickness (int): Font thickness for OCR text labels.
                Defaults to 8.
            conf_text_size (float): Font scale for confidence score text. Defaults to 1.
            conf_text_thickness (int): Font thickness for confidence score text.
                Defaults to 3.
            font_color (Tuple[int, int, int]): BGR color tuple for text rendering.
                Defaults to (0, 0, 255) for red.

        Returns:
            np.ndarray: Annotated image with grid lines and prediction text overlaid.

        Raises:
            ValueError: If image is None, empty, or has invalid dimensions.
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image for annotation")

        # Make a copy of the image to annotate
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

        for (row, col), (all_attempts, _) in predictions.items():
            # Use the best attempt (highest confidence) for annotation
            if not all_attempts:
                continue
            best_attempt = max(all_attempts, key=lambda x: x['confidence'])
            text = best_attempt['text']
            confidence = best_attempt['confidence']
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