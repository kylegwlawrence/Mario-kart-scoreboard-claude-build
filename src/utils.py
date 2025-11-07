"""
Utility functions for logging, file I/O, and helper functions.
"""

import logging
import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logger(
    name: str,
    log_dir: str,
    debug: bool = True,
    console_output: bool = True,
    console_level: Optional[int] = None
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        debug: Enable debug mode (verbose logging)
        console_output: Whether to print to console
        console_level: Log level for console handler (defaults to DEBUG in debug mode, INFO otherwise)

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        if console_level is not None:
            console_handler.setLevel(console_level)
        else:
            console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def save_csv(
    filepath: str,
    data: List[Dict[str, Any]],
    fieldnames: List[str],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save data to a CSV file.

    Args:
        filepath: Path to save CSV file
        data: List of dictionaries to write
        fieldnames: CSV column names
        logger: Logger instance for logging

    Raises:
        IOError: If file writing fails
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        if logger:
            logger.info(f"Saved CSV to {filepath}")
    except IOError as e:
        raise IOError(f"Failed to save CSV to {filepath}: {e}")


def load_csv(
    filepath: str,
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Load data from a CSV file.

    Args:
        filepath: Path to CSV file
        logger: Logger instance for logging

    Returns:
        List of dictionaries

    Raises:
        IOError: If file reading fails
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader)
        if logger:
            logger.info(f"Loaded {len(data)} rows from {filepath}")
        return data
    except IOError as e:
        raise IOError(f"Failed to load CSV from {filepath}: {e}")


def save_json(
    filepath: str,
    data: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save data to a JSON file.

    Args:
        filepath: Path to save JSON file
        data: Dictionary to write
        logger: Logger instance for logging

    Raises:
        IOError: If file writing fails
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2, ensure_ascii=False)
        if logger:
            logger.info(f"Saved JSON to {filepath}")
    except IOError as e:
        raise IOError(f"Failed to save JSON to {filepath}: {e}")


def load_json(
    filepath: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        filepath: Path to JSON file
        logger: Logger instance for logging

    Returns:
        Dictionary loaded from JSON

    Raises:
        IOError: If file reading fails
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
        if logger:
            logger.info(f"Loaded JSON from {filepath}")
        return data
    except IOError as e:
        raise IOError(f"Failed to load JSON from {filepath}: {e}")


def load_valid_player_names(csv_path: str, logger: Optional[logging.Logger] = None) -> set:
    """
    Load valid player names from character_info.csv.

    Args:
        csv_path: Path to character_info.csv
        logger: Logger instance for logging

    Returns:
        Set of valid player names

    Raises:
        IOError: If file reading fails
    """
    try:
        data = load_csv(csv_path, logger)
        names = {row['name'] for row in data if 'name' in row}
        if logger:
            logger.info(f"Loaded {len(names)} valid player names")
        return names
    except (IOError, KeyError) as e:
        raise IOError(f"Failed to load player names from {csv_path}: {e}")
