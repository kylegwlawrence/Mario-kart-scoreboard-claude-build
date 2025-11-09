"""
Main OCR processor orchestrating the full pipeline.
Coordinates preprocessing, OCR, validation, and result storage.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import uuid

from src.config_manager import ConfigManager
from src.preprocessor import PreprocessingPipeline
from src.ocr_engines import OCREngine
from src.annotator import ImageAnnotator
from src.constraint_validator import CellValidator
from src.utils import save_csv, load_valid_player_names, setup_logger


class OCRProcessor:
    """Orchestrates the full OCR pipeline."""

    def __init__(
        self,
        config_path: str,
        debug: bool = None
    ):
        """
        Initialize OCR processor.

        Args:
            config_path: Path to JSON configuration file
            debug: Enable debug logging

        Raises:
            IOError: If config or supporting files cannot be loaded
            ValueError: If config validation fails or debug is not a boolean
        """
        if not isinstance(debug, bool):
            raise ValueError(f"debug parameter must be a boolean, got {type(debug).__name__}")

        self.debug = debug

        # Set up logging
        log_dir = ".logging"
        self.logger = setup_logger(
            name='OCRProcessor',
            log_dir=log_dir,
            debug=debug,
            console_output=True
        )

        self.logger.info("=" * 80)
        self.logger.info("Starting OCR Processor")
        self.logger.info(f"Config path: {config_path}")
        self.logger.info("=" * 80)

        # Load configuration
        try:
            self.config_path = config_path
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

        self.image_annotator = ImageAnnotator(config_manager=self.config_manager, logger=self.logger)

        self.validator = CellValidator(self.valid_player_names, self.logger)
        self.fuzzy_threshold = self.config_manager.get_fuzzy_threshold()

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

        # Generate unique ID for this pipeline run
        run_id = uuid.uuid4().hex[:8]
        self.logger.info(f"Run ID: {run_id}")

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

        # Extract cells using predefined grid bounds
        try:
            cells = self.extract_cells_from_grid(preprocessed_image, columns_to_process=[1, 2, 4])
            self.logger.info(f"Extracted {len(cells)} cells from predefined grid")
        except Exception as e:
            self.logger.error(f"Cell extraction failed: {e}")
            raise

        # Create cell images subdirectory for this run
        cell_images_dir = Path(self.output_paths['cell_images']) / f"{output_filename_prefix}_{run_id}"
        try:
            cell_images_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created cell images directory: {cell_images_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create cell images directory: {e}")
            raise

        # Use full image bounds for annotations
        h, w = preprocessed_image.shape[:2]
        table_bounds = (0, 0, w, h)

        # Process start timestamp
        process_start_time = datetime.now().isoformat()

        # Perform OCR and validation
        predictions, failed_cells = self._process_cells_with_retry(cells, table_bounds, output_filename_prefix, run_id, str(cell_images_dir))

        # Create annotated image
        try:
            annotated_image = self.image_annotator.annotate_image(
                image,
                predictions
            )
            annotated_path = Path(self.output_paths['annotated']) / f"{output_filename_prefix}_{run_id}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.logger.info(f"Written annotated image to: {annotated_path}")
        except Exception as e:
            self.logger.error(f"Failed to create/save annotated image: {e}")
            raise

        # Prepare and save results
        img_h, img_w = image.shape[:2]
        process_end_time = datetime.now().isoformat()
        results = self._prepare_results(
            predictions,
            failed_cells,
            image_path,
            process_start_time,
            process_end_time,
            output_filename_prefix,
            run_id,
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
        table_bounds: Tuple[int, int, int, int],
        filename_prefix: str,
        run_id: str,
        cell_images_dir: str
    ) -> Tuple[Dict[Tuple[int, int], Tuple[str, float, bool, int, Dict, List[str]]], List[Dict[str, Any]]]:
        """
        Process cells with OCR and validation, with retry logic.

        Args:
            cells: Dictionary of (row, col) -> cell_image
            table_bounds: Table bounds for coordinate calculation
            filename_prefix: Prefix for output filenames
            run_id: Unique ID for this pipeline run
            cell_images_dir: Directory to save cell images

        Returns:
            Tuple of (predictions_dict with validation status and pipeline info and cell image paths, failed_cells_list)
        """
        predictions = {}
        failed_cells = []
        retry_attempts = self.config_manager.get_retry_attempts()

        for (row, col), cell_image in sorted(cells.items()):
            cell_text, cell_confidence, passes_validation, retry_attempt_used, chain_config, cell_image_paths = self._process_cell_with_retry(
                cell_image,
                row,
                col,
                retry_attempts,
                predictions,
                filename_prefix,
                run_id,
                cell_images_dir
            )

            if cell_text is not None:
                predictions[(row, col)] = (cell_text, cell_confidence, passes_validation, retry_attempt_used, chain_config, cell_image_paths)
                if not passes_validation:
                    failed_cells.append({
                        'row': row,
                        'col': col,
                        'text': cell_text,
                        'confidence': cell_confidence,
                        'failed_reason': 'Failed validation'
                    })
            else:
                failed_cells.append({
                    'row': row,
                    'col': col,
                    'failed_reason': 'No OCR results extracted'
                })

        return predictions, failed_cells

    def _process_cell_with_retry(
        self,
        cell_image: np.ndarray,
        row: int,
        col: int,
        retry_attempts: int,
        predictions: Optional[Dict[Tuple[int, int], Tuple[str, float, bool, int, Dict, List[str]]]] = None,
        filename_prefix: str = "",
        run_id: str = "",
        cell_images_dir: str = ""
    ) -> Tuple[Optional[str], float, bool, Optional[int], Optional[Dict], List[str]]:
        """
        Process a single cell with all preprocessing chains and select highest confidence result.

        Args:
            cell_image: Cell image
            row: Row index
            col: Column index
            retry_attempts: Maximum retry attempts
            predictions: Dictionary of (row, col) -> (text, confidence, passes_validation, retry_attempt, chain, cell_image_paths) for previous rows
            filename_prefix: Prefix for output filenames
            run_id: Unique ID for this pipeline run
            cell_images_dir: Directory to save cell images

        Returns:
            Tuple of (text, confidence, passes_validation, retry_attempt_used, chain_config, cell_image_paths)
        """
        # Store all OCR attempts with their metadata for later selection
        all_attempts = []
        cell_image_paths = []

        # Get previous row's place for ordering validation (only for place column)
        previous_place = None
        if col == 1 and predictions is not None and row > 0:
            prev_key = (row - 1, 1)
            if prev_key in predictions:
                prev_text, _, prev_passes_validation, _, _, _ = predictions[prev_key]
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
                prev_text, _, prev_passes_validation, _, _, _ = predictions[prev_key]
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

                # Save preprocessed cell image
                if cell_images_dir and filename_prefix and run_id:
                    try:
                        cell_image_filename = f"{filename_prefix}_{run_id}_r{row}_c{col}_attempt{attempt}.jpg"
                        cell_image_path = Path(cell_images_dir) / cell_image_filename
                        cv2.imwrite(str(cell_image_path), preprocessed_cell, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        cell_image_paths.append(str(cell_image_path))
                        if self.logger:
                            self.logger.debug(f"Written cell image to: {cell_image_path}")
                    except Exception as e:
                        if self.logger:
                            self.logger.debug(f"Failed to save cell image ({row}, {col}) attempt {attempt + 1}: {e}")

                # Perform OCR - all predictions will be included in output for analysis
                ocr_results = self.ocr_engine.extract_text(preprocessed_cell)

                if not ocr_results:
                    if self.logger:
                        self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1}: No OCR results")
                    continue

                # Take best result from this attempt
                text, confidence, coords = ocr_results[0]

                # Validate (pass previous_place and previous_score for column ordering validation)
                is_valid, validated_text, error_msg = self.validator.validate_cell(col, text, fuzzy_threshold=self.fuzzy_threshold, previous_place=previous_place, previous_score=previous_score)

                if self.logger:
                    status = "valid" if is_valid else "invalid"
                    self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1}: {status} -> {text} (conf: {confidence:.2f})")

                # Store this attempt result
                all_attempts.append({
                    'text': text,
                    'confidence': confidence,
                    'is_valid': is_valid,
                    'validated_text': validated_text,
                    'attempt': attempt,
                    'chain': chain
                })

            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Cell ({row}, {col}) attempt {attempt + 1} error: {e}")

        # Select best result based on highest confidence
        if all_attempts:
            best_attempt = max(all_attempts, key=lambda x: x['confidence'])
            if self.logger:
                self.logger.debug(f"Cell ({row}, {col}): Selected attempt with highest confidence: {best_attempt['text']} (conf: {best_attempt['confidence']:.2f}), valid: {best_attempt['is_valid']}")
            return (
                best_attempt['text'],
                best_attempt['confidence'],
                best_attempt['is_valid'],
                best_attempt['attempt'],
                best_attempt['chain'],
                cell_image_paths
            )

        if self.logger:
            self.logger.warning(f"Cell ({row}, {col}): No OCR results extracted from any attempt")
        return None, 0.0, False, None, None, cell_image_paths

    def _prepare_results(
        self,
        predictions: Dict[Tuple[int, int], Tuple[str, float, bool, int, Dict, List[str]]],
        failed_cells: List[Dict[str, Any]],
        image_path: str,
        process_start_time: str,
        process_end_time: str,
        filename_prefix: str,
        run_id: str,
        img_width: int,
        img_height: int
    ) -> Dict[str, Any]:
        """
        Prepare and save results to CSV only.

        Args:
            predictions: Dictionary of predictions with validation status, retry attempt, and pipeline config
            failed_cells: List of failed cells with failed_reason
            image_path: Original image path
            process_start_time: Timestamp when processing started
            process_end_time: Timestamp when processing ended
            filename_prefix: Prefix for output files
            run_id: Unique ID for this pipeline run
            img_width: Original image width
            img_height: Original image height

        Returns:
            Dictionary with results metadata
        """
        # Create a mapping of failed cells for quick lookup
        failed_reasons = {}
        for failed_cell in failed_cells:
            key = (failed_cell['row'], failed_cell['col'])
            failed_reasons[key] = failed_cell.get('failed_reason', '')

        # Get primary engine from config
        primary_engine = self.config_manager.get_primary_engine()

        # Prepare CSV data
        csv_data = []
        for (row, col), (text, confidence, passes_validation, retry_attempt_used, chain_config, cell_image_paths) in sorted(predictions.items()):
            cell_x, cell_y, cell_w, cell_h = self.get_cell_coordinates_from_grid(row, col, img_width, img_height)

            # Extract pipeline methods from chain_config
            pipeline_steps = []
            if chain_config and 'methods' in chain_config:
                pipeline_steps = chain_config['methods']

            # Get failed reason if applicable
            failed_reason = failed_reasons.get((row, col), '')

            # Generate unique key that includes retry attempt for tracking all processing attempts
            attempt_num = retry_attempt_used if retry_attempt_used is not None else 0
            unique_key = f"{filename_prefix}_{run_id}_r{row}_c{col}_atmpt{attempt_num}"

            csv_data.append({
                'unique_key': unique_key,
                'row_id': row,
                'column_id': col,
                'predicted_text': text,
                'confidence': f"{confidence:.4f}",
                'passes_validation': str(passes_validation),
                'text_coordinates': f"({cell_x}, {cell_y}, {cell_w}, {cell_h})",
                'original_filepath': image_path,
                'process_start_time': process_start_time,
                'process_end_time': process_end_time,
                'primary_engine': primary_engine,
                'retry_attempt_used': attempt_num,
                'pipeline_steps': json.dumps(pipeline_steps),
                'pipeline_config_path': self.config_path,
                'failed_reason': failed_reason,
                'cell_image_paths': json.dumps(cell_image_paths)
            })

        # Save CSV
        csv_path = Path(self.output_paths['predictions']) / f"{filename_prefix}_{run_id}_predictions.csv"
        try:
            fieldnames = [
                'unique_key', 'row_id', 'column_id', 'predicted_text', 'confidence', 'passes_validation',
                'text_coordinates', 'original_filepath', 'preprocessed_filepath',
                'process_start_time', 'process_end_time', 'primary_engine', 'retry_attempt_used',
                'pipeline_steps', 'pipeline_config_path', 'failed_reason', 'cell_image_paths'
            ]
            save_csv(str(csv_path), csv_data, fieldnames, self.logger)
            self.logger.info(f"Written predictions CSV to: {csv_path}")
        except IOError as e:
            self.logger.error(f"Failed to save CSV: {e}")
            raise

        # Count valid vs invalid predictions
        valid_count = sum(1 for _, _, passes_validation, _, _, _ in predictions.values() if passes_validation)
        invalid_count = len(predictions) - valid_count

        self.logger.info(f"Processing complete: {valid_count} valid, {invalid_count} invalid, {len(failed_cells)} failed out of {12*3} total cells")

        # Return summary metadata
        return {
            'valid_predictions': valid_count,
            'invalid_predictions': invalid_count,
            'failed_cells': len(failed_cells),
            'extracted_cells': len(predictions),
            'output_file': str(csv_path)
        }
