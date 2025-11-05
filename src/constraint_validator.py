"""
Cell validation module for OCR results.
Validates extracted text against constraints.
"""

import logging
from typing import Optional, Set, Tuple
from difflib import SequenceMatcher


class CellValidator:
    """Validates OCR results against table constraints."""

    # Valid place range
    PLACE_MIN = 1
    PLACE_MAX = 12

    # Valid score range
    SCORE_MIN = 1
    SCORE_MAX = 999

    def __init__(
        self,
        valid_player_names: Set[str],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize validator with valid player names.

        Args:
            valid_player_names: Set of valid player names
            logger: Logger instance
        """
        self.valid_player_names = valid_player_names
        self.logger = logger

    def validate_place(self, value: str) -> Tuple[bool, Optional[int], str]:
        """
        Validate place/rank (1-12).

        Args:
            value: Extracted text for place

        Returns:
            Tuple of (is_valid, parsed_value, error_message)
        """
        try:
            # Try to parse as integer
            place = int(value.strip())

            if not (self.PLACE_MIN <= place <= self.PLACE_MAX):
                return False, None, f"Place {place} not in range 1-12"

            return True, place, ""

        except (ValueError, AttributeError):
            return False, None, f"Could not parse place as integer: {value}"

    def validate_score(self, value: str) -> Tuple[bool, Optional[int], str]:
        """
        Validate score (1-999).

        Args:
            value: Extracted text for score

        Returns:
            Tuple of (is_valid, parsed_value, error_message)
        """
        try:
            # Try to parse as integer
            score = int(value.strip())

            if not (self.SCORE_MIN <= score <= self.SCORE_MAX):
                return False, None, f"Score {score} not in range 1-999"

            return True, score, ""

        except (ValueError, AttributeError):
            return False, None, f"Could not parse score as integer: {value}"

    def validate_player_name(
        self,
        value: str,
        fuzzy_threshold: float = 0.8
    ) -> Tuple[bool, Optional[str], str]:
        """
        Validate player name with exact match and fuzzy matching fallback.

        Args:
            value: Extracted text for player name
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-1)

        Returns:
            Tuple of (is_valid, matched_name, error_message)
        """
        if not value:
            return False, None, "Player name is empty"

        value = value.strip()

        # Try exact match first
        if value in self.valid_player_names:
            return True, value, ""

        # Try fuzzy matching
        best_match = None
        best_score = 0

        for name in self.valid_player_names:
            # Calculate similarity ratio
            ratio = SequenceMatcher(None, value.lower(), name.lower()).ratio()

            if ratio > best_score:
                best_score = ratio
                best_match = name

        if best_score >= fuzzy_threshold:
            if self.logger:
                self.logger.debug(
                    f"Fuzzy matched '{value}' to '{best_match}' (score: {best_score:.2f})"
                )
            return True, best_match, ""

        return False, None, f"Player name '{value}' not found (best match: '{best_match}' with score {best_score:.2f})"

    def validate_cell(
        self,
        column: int,
        value: str,
        fuzzy_threshold: float = 0.8
    ) -> Tuple[bool, Optional[any], str]:
        """
        Validate a cell value based on its column.

        Args:
            column: Column index (0=place, 1=name, 2=score)
            value: Extracted text
            fuzzy_threshold: Fuzzy match threshold for names

        Returns:
            Tuple of (is_valid, parsed_value, error_message)

        Raises:
            ValueError: If column index is invalid
        """
        if column == 0:
            return self.validate_place(value)
        elif column == 1:
            return self.validate_player_name(value, fuzzy_threshold)
        elif column == 2:
            return self.validate_score(value)
        else:
            raise ValueError(f"Invalid column index: {column}")
