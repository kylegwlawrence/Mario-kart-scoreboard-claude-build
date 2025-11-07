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
            config_path: Path to JSON config file
            logger: Logger instance

        Raises:
            IOError: If config file cannot be loaded
            ValueError: If config validation fails
        """
        self.logger = logger
        self.config = self._load_and_validate_config(config_path)

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
            'ocr_config',
            'retry_config',
            'character_names_csv'
        ]

        missing_fields = [f for f in required_fields if f not in config]
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")

        # Validate output_paths
        required_output_paths = ['preprocessed', 'annotated', 'predictions', 'logs']
        missing_paths = [p for p in required_output_paths if p not in config['output_paths']]
        if missing_paths:
            raise ValueError(f"Missing required output paths: {missing_paths}")

        # Validate preprocessing chains
        if not config['preprocessing_chains'] or not isinstance(config['preprocessing_chains'], list):
            raise ValueError("preprocessing_chains must be a non-empty list")

        for i, chain in enumerate(config['preprocessing_chains']):
            if 'methods' not in chain or not isinstance(chain['methods'], list):
                raise ValueError(f"Chain {i} must have 'methods' as a list")

        # Validate ocr_config
        if 'engines' not in config['ocr_config'] or not config['ocr_config']['engines']:
            raise ValueError("ocr_config must have at least one engine")

        if 'primary_engine' not in config['ocr_config']:
            raise ValueError("ocr_config must specify primary_engine")

        if config['ocr_config']['primary_engine'] not in config['ocr_config']['engines']:
            raise ValueError("primary_engine must be in the engines list")

        if self.logger:
            self.logger.info(f"Successfully loaded and validated config from {config_path}")

        return config

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

    def get_ocr_config(self) -> Dict[str, Any]:
        """Get OCR configuration."""
        return self.config.get('ocr_config', {})


    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry configuration."""
        return self.config.get('retry_config', {})

    def get_output_paths(self) -> Dict[str, str]:
        """Get output paths."""
        return self.config.get('output_paths', {})

    def get_image_source(self) -> str:
        """Get image source directory."""
        return self.config.get('image_source', '')

    def get_character_names_csv(self) -> str:
        """Get character names CSV path."""
        return self.config.get('character_names_csv', '')

    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """
        Create a default configuration dictionary.

        Returns:
            Default configuration
        """
        return {
            "image_source": "./images",
            "output_paths": {
                "preprocessed": "output/preprocessed_images",
                "annotated": "output/annotated_images",
                "predictions": "output/predictions",
                "logs": ".logging"
            },
            "preprocessing_chains": [
                {
                    "retry_attempt": 0,
                    "methods": [
                        {"method": "grayscale", "parameters": None},
                        {"method": "gaussian_blur", "parameters": {"kernel": (5, 5), "sigmaX": 0, "sigmaY": 0}}
                    ]
                },
                {
                    "retry_attempt": 1,
                    "methods": [
                        {"method": "grayscale", "parameters": None},
                        {"method": "threshold", "parameters": {"threshold": 150, "max_value": 255}}
                    ]
                }
            ],
            "ocr_config": {
                "engines": ["paddleocr", "tesseract", "easyocr"],
                "primary_engine": "paddleocr"
            },
            "retry_config": {
                "max_attempts": 3,
                "retry_on_low_confidence": True,
                "retry_strategies": ["preprocessing", "ocr_engine"]
            },
            "character_names_csv": "character_info.csv"
        }
