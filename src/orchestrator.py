"""
Main OCR processor orchestrating the full pipeline.
Coordinates preprocessing, OCR, validation, and result storage.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.config_manager import ConfigManager
from src.preprocessor import PreprocessingPipeline
from src.ocr_engines import OCREngine
from src.table_detector import TableDetector
from src.constraint_validator import CellValidator
from src.utils import save_csv, save_json, load_valid_player_names, setup_logger


class OCRProcessor:
    """Orchestrates the full OCR pipeline."""

    def __init__(
        self,
        config_path: str,
        debug: bool = False
    ):
        """
        Initialize OCR processor.

        Args:
            config_path: Path to JSON configuration file
            debug: Enable debug logging

        Raises:
            IOError: If config or supporting files cannot be loaded
            ValueError: If config validation fails
        """
        self.debug = debug

        # Set up logging
        log_dir = ".logging"
        self.logger = setup_logger(
            name='OCRProcessor',
            log_dir=log_dir,
            debug=debug,
            console_output=True,
            console_level=logging.DEBUG if debug else logging.WARNING
        )

        self.logger.info("=" * 80)
        self.logger.info("Starting OCR Processor")
        self.logger.info(f"Config path: {config_path}")
        self.logger.info("=" * 80)

        # Load configuration
        try:
            self.config_manager = ConfigManager(config_path, self.logger)
            self.config = self.config_manager.config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

        # Load output directories (assumed to be already created)
        try:
            self.output_paths = self.config_manager.get_output_paths()
            self.logger.info(f"Output directories: {self.output_paths}")
        except Exception as e:
            self.logger.error(f"Failed to load output directories: {e}")
            raise

        # Load valid player names
        try:
            character_csv = self.config_manager.get_character_names_csv()
            self.valid_player_names = load_valid_player_names(character_csv, self.logger)
        except IOError as e:
            self.logger.error(f"Failed to load player names: {e}")
            raise

        # Initialize components
        self.preprocessing = PreprocessingPipeline(self.logger)

        # get the primary OCR engine from the config files and init the OCR engine
        primary_engine = self.config_manager.get_primary_engine()
        self.ocr_engine = OCREngine(
            config_manager=self.config_manager,
            primary_engine=primary_engine,
            logger=self.logger
        )

        self.table_detector = TableDetector(config_manager=self.config_manager, logger=self.logger)

        self.validator = CellValidator(self.valid_player_names, self.logger)

        self.logger.info("OCR Processor initialized successfully")

    def process_image(
        self,
        image_path: str,
        output_filename_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single image through the full OCR pipeline.

        Args:
            image_path: Path to input image
            output_filename_prefix: Optional prefix for output files (defaults to image filename)

        Returns:
            Dictionary with results and metadata

        Raises:
            IOError: If image cannot be loaded or files cannot be saved
            Exception: If processing fails critically
        """
        self.logger.info(f"\nProcessing image: {image_path}")

        # Load image
        try:
            # Ensure path is a string
            image_path_str = str(image_path)
            image = cv2.imread(image_path_str)
            if image is None:
                raise IOError(f"cv2.imread failed for {image_path_str}. File may be corrupted or unsupported format.")
            self.logger.info(f"Loaded image: {image.shape}")
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            raise

        # Set output prefix
        if output_filename_prefix is None:
            output_filename_prefix = Path(image_path).stem

        # Preprocess image
        try:
            preprocessed_image, preprocessing_methods = self._preprocess_image(image)
            self.logger.info(f"Preprocessing complete: {len(preprocessing_methods)} methods applied")
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise

        # Save preprocessed image
        preprocessed_path = Path(self.output_paths['preprocessed']) / f"{output_filename_prefix}_preprocessed.png"
        try:
            self.preprocessing.save_preprocessed_image(preprocessed_image, str(preprocessed_path))
        except IOError as e:
            self.logger.error(f"Failed to save preprocessed image: {e}")
            raise

        # Extract cells using predefined grid bounds
        try:
            cells = self.extract_cells_from_grid(preprocessed_image, columns_to_process=[1, 2, 4])
            self.logger.info(f"Extracted {len(cells)} cells from predefined grid")
        except Exception as e:
            self.logger.error(f"Cell extraction failed: {e}")
            raise

        # Use full image bounds for annotations
        h, w = preprocessed_image.shape[:2]
        table_bounds = (0, 0, w, h)

        # Process start timestamp
        process_start_time = datetime.now().isoformat()

        # Perform OCR and validation
        predictions, failed_cells = self._process_cells_with_retry(cells, table_bounds)

        # Create annotated image
        try:
            annotated_image = self.table_detector.annotate_image(
                image,
                predictions,
                table_bounds
            )
            annotated_path = Path(self.output_paths['annotated']) / f"{output_filename_prefix}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.logger.info(f"Saved annotated image: {annotated_path}")
        except Exception as e:
            self.logger.error(f"Failed to create/save annotated image: {e}")
            raise

        # Prepare and save results
        img_h, img_w = image.shape[:2]
        results = self._prepare_results(
            predictions,
            failed_cells,
            image_path,
            str(preprocessed_path),
            preprocessing_methods,
            process_start_time,
            output_filename_prefix,
            img_w,
            img_h
        )

        return results

    def _preprocess_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply preprocessing to image using the initial preprocessing chain (retry_attempt=0).

        Args:
            image: Input image

        Returns:
            Tuple of (preprocessed_image, list_of_applied_methods)
        """
        # Get the initial preprocessing chain (attempt 0)
        chain = self.config_manager.get_preprocessing_chain_by_retry_attempt(0)

        if chain is None:
            # Fallback: get first chain if attempt 0 not found
            chains = self.config_manager.get_preprocessing_chains()
            if not chains:
                raise ValueError("No preprocessing chains configured")
            chain = chains[0]

        methods = chain.get('methods', [])
        preprocessed, applied_methods = self.preprocessing.apply_chain(image, methods)
        return preprocessed, applied_methods

    def extract_cells_from_grid(
        self,
        image: np.ndarray,
        columns_to_process: Optional[List[int]] = None
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Extract cells from image using predefined grid bounds.

        Args:
            image: Preprocessed image
            columns_to_process: List of column indices to extract (default: [1, 2, 4])

        Returns:
            Dictionary with (row, col) keys and cell images as values
        """
        if columns_to_process is None:
            columns_to_process = [1, 2, 4]

        cells = {}
        img_height, img_width = image.shape[:2]

        num_rows = self.config_manager.get_num_rows()

        for row in range(num_rows):
            for col in columns_to_process:
                try:
                    # Get percentage bounds from config_manager
                    left_pct, top_pct, right_pct, bottom_pct = self.config_manager.get_cell_bounds(row, col)

                    # Convert percentage bounds to pixel coordinates
                    x = int(left_pct * img_width)
                    y = int(top_pct * img_height)
                    width = int((right_pct - left_pct) * img_width)
                    height = int((bottom_pct - top_pct) * img_height)

                    # Extract cell image
                    cell_image = image[y:y+height, x:x+width]

                    if cell_image.size > 0:
                        cells[(row, col)] = cell_image
                    else:
                        self.logger.warning(f"Cell ({row}, {col}): Empty region extracted")

                except Exception as e:
                    self.logger.error(f"Failed to extract cell ({row}, {col}): {e}")

        return cells

    def get_cell_coordinates_from_grid(
        self,
        row: int,
        col: int,
        img_width: int,
        img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Get pixel coordinates of a cell using grid bounds.

        Args:
            row: Row index (0-11)
            col: Column index (0-4)
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (x, y, width, height) in pixel coordinates
        """
        try:
            left_pct, top_pct, right_pct, bottom_pct = self.config_manager.get_cell_bounds(row, col)

            x = int(left_pct * img_width)
            y = int(top_pct * img_height)
            width = int((right_pct - left_pct) * img_width)
            height = int((bottom_pct - top_pct) * img_height)

            return x, y, width, height
        except Exception as e:
            self.logger.error(f"Failed to get coordinates for cell ({row}, {col}): {e}")
            return 0, 0, 0, 0

    def _process_cells_with_retry(
        self,
        cells: Dict[Tuple[int, int], np.ndarray],
        table_bounds: Tuple[int, int, int, int]
    ) -> Tuple[Dict[Tuple[int, int], Tuple[str, float, bool]], List[Dict[str, Any]]]:
        """
        Process cells with OCR and validation, with retry logic.

        Args:
            cells: Dictionary of (row, col) -> cell_image
            table_bounds: Table bounds for coordinate calculation

        Returns:
            Tuple of (predictions_dict with validation status, failed_cells_list)
        """
        predictions = {}
        failed_cells = []
        retry_attempts = self.config_manager.get_retry_attempts()

        for (row, col), cell_image in sorted(cells.items()):
            cell_text, cell_confidence, passes_validation = self._process_cell_with_retry(
                cell_image,
                row,
                col,
                retry_attempts,
                predictions
            )

            if cell_text is not None:
                predictions[(row, col)] = (cell_text, cell_confidence, passes_validation)
                if not passes_validation:
                    failed_cells.append({
                        'row': row,
                        'col': col,
                        'text': cell_text,
                        'confidence': cell_confidence,
                        'reason': 'Failed validation'
                    })
            else:
                failed_cells.append({
                    'row': row,
                    'col': col,
                    'reason': 'No OCR results extracted'
                })

        return predictions, failed_cells

    def _process_cell_with_retry(
        self,
        cell_image: np.ndarray,
        row: int,
        col: int,
        retry_attempts: int,
        predictions: Optional[Dict[Tuple[int, int], Tuple[str, float, bool]]] = None
    ) -> Tuple[Optional[str], float, bool]:
        """
        Process a single cell with retry logic using attempt-specific preprocessing chains.

        Args:
            cell_image: Cell image
            row: Row index
            col: Column index
            retry_attempts: Maximum retry attempts
            predictions: Dictionary of (row, col) -> (text, confidence, passes_validation) for previous rows

        Returns:
            Tuple of (text, confidence, passes_validation) where passes_validation is True only if text is valid
        """
        last_extracted_text = None
        last_confidence = 0.0

        # Get previous row's place for ordering validation (only for place column)
        previous_place = None
        if col == 1 and predictions is not None and row > 0:
            prev_key = (row - 1, 1)
            if prev_key in predictions:
                prev_text, _, prev_passes_validation = predictions[prev_key]
                # Only use previous place if it passed validation
                if prev_passes_validation:
                    try:
                        previous_place = int(prev_text)
                    except (ValueError, TypeError):
                        pass

        # Get previous row's score for ordering validation (only for score column)
        previous_score = None
        if col == 4 and predictions is not None and row > 0:
            prev_key = (row - 1, 4)
            if prev_key in predictions:
                prev_text, _, prev_passes_validation = predictions[prev_key]
                # Only use previous score if it passed validation
                if prev_passes_validation:
                    try:
                        previous_score = int(prev_text)
                    except (ValueError, TypeError):
                        pass

        for attempt in range(retry_attempts):
            try:
                # Get preprocessing chain for this retry attempt
                chain = self.config_manager.get_preprocessing_chain_by_retry_attempt(attempt)

                # Skip attempt if no matching chain found
                if chain is None:
                    if self.logger:
                        self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1}: No preprocessing chain found, skipping")
                    continue

                # Apply preprocessing specific to this retry attempt
                methods = chain.get('methods', [])
                preprocessed_cell, applied_methods = self.preprocessing.apply_chain(cell_image, methods)

                if self.logger:
                    self.logger.debug(
                        f"Cell ({row}, {col}) attempt {attempt + 1}: "
                        f"Using preprocessing chain retry_attempt={chain.get('retry_attempt', '?')} "
                        f"({len(applied_methods)} methods)"
                    )

                # Perform OCR - all predictions will be included in output for analysis
                ocr_results = self.ocr_engine.extract_text(preprocessed_cell)

                if not ocr_results:
                    if self.logger:
                        self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1}: No OCR results")
                    continue

                # Take best result
                text, confidence, coords = ocr_results[0]
                last_extracted_text = text
                last_confidence = confidence

                # Validate (pass previous_place and previous_score for column ordering validation)
                is_valid, validated_text, error_msg = self.validator.validate_cell(col, text, previous_place=previous_place, previous_score=previous_score)

                if is_valid:
                    if self.logger:
                        self.logger.debug(f"Cell ({row}, {col}): Valid -> {validated_text} (conf: {confidence:.2f})")
                    return validated_text, confidence, True
                else:
                    if self.logger:
                        self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1}: {error_msg}")
                    # Store the extracted text even if it failed validation
                    if last_extracted_text is None:
                        last_extracted_text = text

            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1} error: {e}")

        if self.logger and last_extracted_text is not None:
            self.logger.warning(f"Cell ({row}, {col}): Failed validation after {retry_attempts} attempts, last extracted text: {last_extracted_text}, confidence: {last_confidence}")

        # Return the last extracted text with validation status as False
        if last_extracted_text is not None:
            return last_extracted_text, last_confidence, False

        return None, 0.0, False

    def _prepare_results(
        self,
        predictions: Dict[Tuple[int, int], Tuple[str, float, bool]],
        failed_cells: List[Dict[str, Any]],
        image_path: str,
        preprocessed_path: str,
        preprocessing_methods: List[str],
        process_start_time: str,
        filename_prefix: str,
        img_width: int,
        img_height: int
    ) -> Dict[str, Any]:
        """
        Prepare and save results to CSV and JSON.

        Args:
            predictions: Dictionary of predictions with validation status (text, confidence, passes_validation)
            failed_cells: List of failed cells
            image_path: Original image path
            preprocessed_path: Preprocessed image path
            preprocessing_methods: List of preprocessing methods applied
            process_start_time: Timestamp when processing started
            filename_prefix: Prefix for output files
            img_width: Original image width
            img_height: Original image height

        Returns:
            Dictionary with results metadata
        """
        # Prepare CSV data
        csv_data = []
        for (row, col), (text, confidence, passes_validation) in sorted(predictions.items()):
            cell_x, cell_y, cell_w, cell_h = self.get_cell_coordinates_from_grid(row, col, img_width, img_height)

            csv_data.append({
                'row_id': row,
                'column_id': col,
                'predicted_text': text,
                'confidence': f"{confidence:.4f}",
                'passes_validation': str(passes_validation),
                'text_coordinates': f"({cell_x}, {cell_y}, {cell_w}, {cell_h})",
                'original_filepath': image_path,
                'preprocessed_filepath': preprocessed_path,
                'process_start_timestamp': process_start_time,
                'preprocessing_methods': '; '.join(preprocessing_methods)
            })

        # Save CSV
        csv_path = Path(self.output_paths['predictions']) / f"{filename_prefix}_predictions.csv"
        try:
            fieldnames = [
                'row_id', 'column_id', 'predicted_text', 'confidence', 'passes_validation',
                'text_coordinates', 'original_filepath', 'preprocessed_filepath',
                'process_start_timestamp', 'preprocessing_methods'
            ]
            save_csv(str(csv_path), csv_data, fieldnames, self.logger)
        except IOError as e:
            self.logger.error(f"Failed to save CSV: {e}")
            raise

        # Save JSON with configuration and metadata
        # Count valid vs invalid predictions
        valid_count = sum(1 for _, _, passes_validation in predictions.values() if passes_validation)
        invalid_count = len(predictions) - valid_count

        json_data = {
            'filename_prefix': filename_prefix,
            'image_path': image_path,
            'preprocessed_path': preprocessed_path,
            'process_start_time': process_start_time,
            'process_end_time': datetime.now().isoformat(),
            'total_cells': 12 * 3,
            'extracted_cells': len(predictions),
            'valid_predictions': valid_count,
            'invalid_predictions': invalid_count,
            'failed_cells': len(failed_cells),
            'preprocessing_config': {
                'methods_applied': preprocessing_methods,
                'configuration': self.config_manager.get_preprocessing_chains()[0]
            },
            'ocr_config': self.config_manager.get_ocr_config(),
            'failed_cells_details': failed_cells
        }

        json_path = Path(self.output_paths['predictions']) / f"{filename_prefix}_metadata.json"
        try:
            save_json(str(json_path), json_data, self.logger)
        except IOError as e:
            self.logger.error(f"Failed to save JSON: {e}")
            raise

        self.logger.info(f"Processing complete: {valid_count} valid, {invalid_count} invalid, {len(failed_cells)} failed out of {12*3} total cells")

        return json_data
