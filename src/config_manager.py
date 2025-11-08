"""
Configuration management for OCR pipeline.
Loads and validates JSON configuration files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils import load_json


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize config manager and load configuration.

        Args:
            config_path: Path to JSON pipeline config file
            logger: Logger instance

        Raises:
            IOError: If config file cannot be loaded
            ValueError: If config validation fails
        """
        self.logger = logger
        self.config = self._load_and_validate_config(config_path)

        # Load grid and OCR engine configurations from fixed locations
        config_dir = Path(config_path).parent.parent
        self.grid_config = self._load_grid_config(str(config_dir / 'grid.json'))
        self.ocr_engines_config = self._load_ocr_engines_config(str(config_dir / 'ocr_engines.json'))

    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and validate configuration from JSON file.

        Args:
            config_path: Path to JSON config file

        Returns:
            Validated configuration dictionary

        Raises:
            IOError: If config file cannot be loaded
            ValueError: If config validation fails
        """
        try:
            config = load_json(config_path, self.logger)
        except IOError as e:
            raise IOError(f"Failed to load config from {config_path}: {e}")

        # Validate required fields
        required_fields = [
            'image_source',
            'output_paths',
            'preprocessing_chains',
            'primary_engine',
            'retry_attempts'
        ]

        missing_fields = [f for f in required_fields if f not in config]
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")

        # Validate output_paths
        required_output_paths = ['preprocessed', 'annotated', 'predictions', 'cell_images', 'logs']
        missing_paths = [p for p in required_output_paths if p not in config['output_paths']]
        if missing_paths:
            raise ValueError(f"Missing required output paths: {missing_paths}")

        # Validate preprocessing chains
        if not config['preprocessing_chains'] or not isinstance(config['preprocessing_chains'], list):
            raise ValueError("preprocessing_chains must be a non-empty list")

        for i, chain in enumerate(config['preprocessing_chains']):
            if 'methods' not in chain or not isinstance(chain['methods'], list):
                raise ValueError(f"Chain {i} must have 'methods' as a list")

        # Validate primary_engine
        if not isinstance(config['primary_engine'], str) or not config['primary_engine'].strip():
            raise ValueError("primary_engine must be a non-empty string")

        if self.logger:
            self.logger.info(f"Successfully loaded and validated config from {config_path}")

        return config

    def _load_grid_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load grid configuration from JSON file.

        Args:
            config_path: Path to grid.json

        Returns:
            Grid configuration dictionary

        Raises:
            IOError: If config file cannot be loaded
            ValueError: If config validation fails
        """
        try:
            grid_config = load_json(config_path, self.logger)
        except IOError as e:
            raise IOError(f"Failed to load grid config from {config_path}: {e}")

        # Validate grid configuration
        required_fields = ['num_rows', 'num_columns', 'column_bounds', 'row_bounds']
        missing_fields = [f for f in required_fields if f not in grid_config]
        if missing_fields:
            raise ValueError(f"Grid config missing required fields: {missing_fields}")

        if not isinstance(grid_config['column_bounds'], list) or len(grid_config['column_bounds']) == 0:
            raise ValueError("column_bounds must be a non-empty list")

        if not isinstance(grid_config['row_bounds'], list) or len(grid_config['row_bounds']) == 0:
            raise ValueError("row_bounds must be a non-empty list")

        if self.logger:
            self.logger.info(f"Successfully loaded grid config from {config_path}")

        return grid_config

    def _load_ocr_engines_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load OCR engines configuration from JSON file.

        Args:
            config_path: Path to ocr_engines.json

        Returns:
            OCR engines configuration dictionary

        Raises:
            IOError: If config file cannot be loaded
            ValueError: If config validation fails
        """
        try:
            ocr_engines_config = load_json(config_path, self.logger)
        except IOError as e:
            raise IOError(f"Failed to load OCR engines config from {config_path}: {e}")

        # Validate OCR engines configuration
        required_fields = ['character_names_csv', 'engines']
        missing_fields = [f for f in required_fields if f not in ocr_engines_config]
        if missing_fields:
            raise ValueError(f"OCR engines config missing required fields: {missing_fields}")

        if not isinstance(ocr_engines_config['engines'], dict) or not ocr_engines_config['engines']:
            raise ValueError("engines must be a non-empty dictionary")

        if self.logger:
            self.logger.info(f"Successfully loaded OCR engines config from {config_path}")

        return ocr_engines_config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def get_nested(self, keys: List[str], default: Any = None) -> Any:
        """
        Get nested configuration value.

        Args:
            keys: List of keys to traverse (e.g., ['output_paths', 'preprocessed'])
            default: Default value if key path not found

        Returns:
            Configuration value
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def get_preprocessing_chains(self) -> List[Dict[str, Any]]:
        """Get all preprocessing chains."""
        return self.config.get('preprocessing_chains', [])

    def get_primary_engine(self) -> str:
        """Get the primary OCR engine name."""
        return self.config.get('primary_engine', 'tesseract')

    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """
        Get configuration parameters for a specific OCR engine.

        Args:
            engine_name: Name of the OCR engine (e.g., 'paddleocr', 'tesseract', 'easyocr')

        Returns:
            Engine-specific configuration dictionary
        """
        return self.ocr_engines_config.get('engines', {}).get(engine_name, {})

    def get_ocr_config(self) -> Dict[str, Any]:
        """
        Get OCR configuration for output/metadata purposes.
        Returns a dict with primary_engine and available engines from ocr_engines_config.

        Returns:
            Dictionary with ocr configuration
        """
        return {
            'primary_engine': self.get_primary_engine(),
            'engines': list(self.ocr_engines_config.get('engines', {}).keys())
        }

    def get_retry_attempts(self) -> int:
        """Get maximum number of retry attempts."""
        return self.config.get('retry_attempts', 3)

    def get_preprocessing_chain_by_retry_attempt(self, attempt: int) -> Optional[Dict[str, Any]]:
        """
        Get preprocessing chain for a specific retry attempt.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Preprocessing chain configuration or None if not found
        """
        chains = self.get_preprocessing_chains()
        for chain in chains:
            if chain.get('retry_attempt') == attempt:
                return chain

        # If exact attempt not found, return None (skip this attempt)
        return None

    def get_output_paths(self) -> Dict[str, str]:
        """Get output paths."""
        return self.config.get('output_paths', {})

    def get_image_source(self) -> str:
        """Get image source directory."""
        return self.config.get('image_source', '')

    def get_character_names_csv(self) -> str:
        """Get character names CSV path from OCR engines config."""
        return self.ocr_engines_config.get('character_names_csv', 'data/character_info.csv')

    def get_table_bounds(self) -> Dict[str, Any]:
        """Get table bounds configuration from grid config."""
        return self.grid_config

    def get_num_rows(self) -> int:
        """Get number of rows in the table grid."""
        return self.grid_config.get('num_rows', 12)

    def get_num_columns(self) -> int:
        """Get number of columns in the table grid."""
        return self.grid_config.get('num_columns', 5)

    def get_column_bounds(self) -> List[List[float]]:
        """Get column bounds as list of [left, right] percentage pairs."""
        return self.grid_config.get('column_bounds', [])

    def get_row_bounds(self) -> List[List[float]]:
        """Get row bounds as list of [top, bottom] percentage pairs."""
        return self.grid_config.get('row_bounds', [])

    def get_cell_bounds(self, row: int, col: int) -> tuple:
        """
        Get the bounding box coordinates for a specific cell as percentages.

        Args:
            row: Row index
            col: Column index

        Returns:
            Tuple of (left, top, right, bottom) as decimal percentages (0.0-1.0)

        Raises:
            ValueError: If cell position is invalid
        """
        num_rows = self.get_num_rows()
        num_columns = self.get_num_columns()
        column_bounds = self.get_column_bounds()
        row_bounds = self.get_row_bounds()

        if not (0 <= row < num_rows and 0 <= col < num_columns):
            raise ValueError(f"Invalid cell position: row={row}, col={col}")

        col_left, col_right = column_bounds[col]
        row_top, row_bottom = row_bounds[row]

        return (col_left, row_top, col_right, row_bottom)

    def get_fuzzy_threshold(self) -> float:
        """
        Get fuzzy matching threshold for player name validation.

        Returns:
            Fuzzy threshold value (0.0-1.0), defaults to 0.8
        """
        return self.config.get('fuzzy_threshold', 0.8)

