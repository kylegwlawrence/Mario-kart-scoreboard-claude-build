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
from src.utils import save_csv, save_json, load_valid_player_names


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
        self.logger = self._setup_logger(log_dir, debug)

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

        ocr_config = self.config_manager.get_ocr_config()
        # get the primary OCR engine from the config files and init the OCR engine
        self.ocr_engine = OCREngine(
            primary_engine=ocr_config.get('primary_engine', 'paddleocr'),
            logger=self.logger
        )

        table_config = self.config_manager.get_table_detection_config()
        self.table_detector = TableDetector(
            enabled=table_config.get('enabled', True),
            method=table_config.get('method', 'contour'),
            logger=self.logger
        )

        self.validator = CellValidator(self.valid_player_names, self.logger)

        self.logger.info("OCR Processor initialized successfully")

    def _setup_logger(self, log_dir: str, debug: bool) -> logging.Logger:
        """Set up logger for the processor."""
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger('OCRProcessor')
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        log_file = Path(log_dir) / f'processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

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

        # Detect table bounds
        try:
            table_bounds = self.table_detector.find_table_bounds(preprocessed_image)
            self.logger.info(f"Table bounds detected: {table_bounds}")
        except Exception as e:
            self.logger.warning(f"Table detection failed: {e}")
            # Use full image bounds
            h, w = preprocessed_image.shape[:2]
            table_bounds = (0, 0, w, h)

        # Extract table cells
        try:
            cells = self.table_detector.extract_cells(preprocessed_image, table_bounds)
            self.logger.info(f"Extracted {len(cells)} cells from table")
        except Exception as e:
            self.logger.error(f"Cell extraction failed: {e}")
            raise

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
        results = self._prepare_results(
            predictions,
            failed_cells,
            image_path,
            str(preprocessed_path),
            preprocessing_methods,
            process_start_time,
            output_filename_prefix
        )

        return results

    def _preprocess_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply preprocessing to image using configured chain.

        Args:
            image: Input image

        Returns:
            Tuple of (preprocessed_image, list_of_applied_methods)
        """
        preprocessing_chains = self.config_manager.get_preprocessing_chains()

        if not preprocessing_chains:
            raise ValueError("No preprocessing chains configured")

        # Use first chain by default
        chain = preprocessing_chains[0]
        methods = chain.get('methods', [])

        preprocessed, applied_methods = self.preprocessing.apply_chain(image, methods)
        return preprocessed, applied_methods

    def _process_cells_with_retry(
        self,
        cells: Dict[Tuple[int, int], np.ndarray],
        table_bounds: Tuple[int, int, int, int]
    ) -> Tuple[Dict[Tuple[int, int], Tuple[str, float]], List[Dict[str, Any]]]:
        """
        Process cells with OCR and validation, with retry logic.

        Args:
            cells: Dictionary of (row, col) -> cell_image
            table_bounds: Table bounds for coordinate calculation

        Returns:
            Tuple of (predictions_dict, failed_cells_list)
        """
        predictions = {}
        failed_cells = []
        retry_config = self.config_manager.get_retry_config()
        ocr_config = self.config_manager.get_ocr_config()
        max_attempts = retry_config.get('max_attempts', 3)
        confidence_threshold = ocr_config.get('confidence_threshold', 0.5)

        for (row, col), cell_image in sorted(cells.items()):
            cell_text, cell_confidence = self._process_cell_with_retry(
                cell_image,
                row,
                col,
                max_attempts,
                confidence_threshold
            )

            if cell_text is not None:
                predictions[(row, col)] = (cell_text, cell_confidence)
            else:
                failed_cells.append({
                    'row': row,
                    'col': col,
                    'reason': 'Failed validation after all retries'
                })

        return predictions, failed_cells

    def _process_cell_with_retry(
        self,
        cell_image: np.ndarray,
        row: int,
        col: int,
        max_attempts: int,
        confidence_threshold: float
    ) -> Tuple[Optional[str], float]:
        """
        Process a single cell with retry logic.

        Args:
            cell_image: Cell image
            row: Row index
            col: Column index
            max_attempts: Maximum retry attempts
            confidence_threshold: Minimum confidence threshold

        Returns:
            Tuple of (validated_text, confidence) or (None, 0.0) if validation failed
        """
        for attempt in range(max_attempts):
            try:
                # Perform OCR
                ocr_results = self.ocr_engine.extract_text(cell_image, confidence_threshold)

                if not ocr_results:
                    if self.logger:
                        self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1}: No OCR results")
                    continue

                # Take best result
                text, confidence, coords = ocr_results[0]

                # Validate
                is_valid, validated_text, error_msg = self.validator.validate_cell(col, text)

                if is_valid:
                    if self.logger:
                        self.logger.debug(f"Cell ({row}, {col}): Valid -> {validated_text} (conf: {confidence:.2f})")
                    return validated_text, confidence
                else:
                    if self.logger:
                        self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1}: {error_msg}")

            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1} error: {e}")

        if self.logger:
            self.logger.warning(f"Cell ({row}, {col}): Failed validation after {max_attempts} attempts")

        return None, 0.0

    def _prepare_results(
        self,
        predictions: Dict[Tuple[int, int], Tuple[str, float]],
        failed_cells: List[Dict[str, Any]],
        image_path: str,
        preprocessed_path: str,
        preprocessing_methods: List[str],
        process_start_time: str,
        filename_prefix: str
    ) -> Dict[str, Any]:
        """
        Prepare and save results to CSV and JSON.

        Args:
            predictions: Dictionary of valid predictions
            failed_cells: List of failed cells
            image_path: Original image path
            preprocessed_path: Preprocessed image path
            preprocessing_methods: List of preprocessing methods applied
            process_start_time: Timestamp when processing started
            filename_prefix: Prefix for output files

        Returns:
            Dictionary with results metadata
        """
        # Prepare CSV data
        csv_data = []
        for (row, col), (text, confidence) in sorted(predictions.items()):
            cell_x, cell_y, cell_w, cell_h = self.table_detector.get_cell_coordinates(row, col, (0, 0, 1920, 1080))

            csv_data.append({
                'row_id': row,
                'column_id': col,
                'predicted_text': text,
                'confidence': f"{confidence:.4f}",
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
                'row_id', 'column_id', 'predicted_text', 'confidence',
                'text_coordinates', 'original_filepath', 'preprocessed_filepath',
                'process_start_timestamp', 'preprocessing_methods'
            ]
            save_csv(str(csv_path), csv_data, fieldnames, self.logger)
        except IOError as e:
            self.logger.error(f"Failed to save CSV: {e}")
            raise

        # Save JSON with configuration and metadata
        json_data = {
            'filename_prefix': filename_prefix,
            'image_path': image_path,
            'preprocessed_path': preprocessed_path,
            'process_start_time': process_start_time,
            'process_end_time': datetime.now().isoformat(),
            'total_cells': 12 * 3,
            'valid_predictions': len(predictions),
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

        self.logger.info(f"Processing complete: {len(predictions)}/{12*3} cells valid")

        return json_data
