"""
Cell validation module for OCR results.

Validates extracted text against table constraints for Mario Kart scoreboard data.
Provides validation for place rankings, player names, and scores with support for
fuzzy matching and sequential ordering constraints.
"""

import logging
from typing import Optional, Set, Tuple
from difflib import SequenceMatcher


class CellValidator:
    """
    Validates OCR results against table constraints.

    Handles validation of individual cells in a Mario Kart scoreboard table,
    including place rankings (1-12), player names with fuzzy matching, and
    race scores (1-999). Supports sequential ordering constraints to ensure
    places decrease and scores decrease down the table.

    Attributes:
        PLACE_MIN: Minimum valid place value (1)
        PLACE_MAX: Maximum valid place value (12)
        SCORE_MIN: Minimum valid score value (1)
        SCORE_MAX: Maximum valid score value (999)
        valid_player_names: Set of acceptable player names
        logger: Optional logger for debug information
    """

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
            valid_player_names: Set of valid player names for fuzzy matching lookup
            logger: Optional logger instance for debug messages

        Example:
            >>> validator = CellValidator({'Alice', 'Bob', 'Charlie'})
            >>> is_valid, value, error = validator.validate_player_name('Alice')
            >>> print(is_valid)
            True
        """
        self.valid_player_names = valid_player_names
        self.logger = logger

    def validate_place(
        self,
        value: str,
        previous_place: Optional[int] = None
    ) -> Tuple[bool, Optional[int], str]:
        """
        Validate place/rank (1-12) with optional ordering constraint.

        Validates that a place value is a valid integer in the range 1-12.
        If a previous place is provided, enforces that the current place must be
        greater than or equal to the previous place (places can only stay same or increase).

        Args:
            value: Extracted text for place (should be convertible to integer)
            previous_place: Previous row's place value for ordering validation.
                           If provided, current place must be >= previous place.

        Returns:
            Tuple of (is_valid, parsed_value, error_message):
                - is_valid (bool): Whether the place passed validation
                - parsed_value (int | None): The integer place value if valid, None otherwise
                - error_message (str): Descriptive error message if invalid, empty string if valid

        Example:
            >>> validator = CellValidator(set())
            >>> is_valid, place, error = validator.validate_place('5')
            >>> print(is_valid, place)
            True 5
            >>> is_valid, place, error = validator.validate_place('3', previous_place=5)
            >>> print(is_valid, error)
            False Place 3 is less than previous row's place 5
        """
        try:
            # Try to parse as integer
            place = int(value.strip())

            # Validate place is within valid range
            if not (self.PLACE_MIN <= place <= self.PLACE_MAX):
                return False, None, f"Place {place} not in range {self.PLACE_MIN}-{self.PLACE_MAX}"

            # Check ordering constraint if previous_place is provided
            if previous_place is not None and place < previous_place:
                return False, None, f"Place {place} is less than previous row's place {previous_place}"

            return True, place, ""

        except (ValueError, AttributeError):
            return False, None, f"Could not parse place as integer: {value}"

    def validate_score(
        self,
        value: str,
        previous_score: Optional[int] = None
    ) -> Tuple[bool, Optional[int], str]:
        """
        Validate score (1-999) with optional ordering constraint.

        Validates that a score value is a valid integer in the range 1-999.
        If a previous score is provided, enforces that the current score must be
        less than or equal to the previous score (scores must be non-increasing down the table).

        Args:
            value: Extracted text for score (should be convertible to integer)
            previous_score: Previous row's score value for ordering validation.
                           If provided, current score must be <= previous score.

        Returns:
            Tuple of (is_valid, parsed_value, error_message):
                - is_valid (bool): Whether the score passed validation
                - parsed_value (int | None): The integer score value if valid, None otherwise
                - error_message (str): Descriptive error message if invalid, empty string if valid

        Example:
            >>> validator = CellValidator(set())
            >>> is_valid, score, error = validator.validate_score('250')
            >>> print(is_valid, score)
            True 250
            >>> is_valid, score, error = validator.validate_score('300', previous_score=250)
            >>> print(is_valid, error)
            False Score 300 is greater than previous row's score 250
        """
        try:
            # Try to parse as integer
            score = int(value.strip())

            # Validate score is within valid rang
            if not (self.SCORE_MIN <= score <= self.SCORE_MAX):
                return False, None, f"Score {score} not in range {self.SCORE_MIN}-{self.SCORE_MAX}"

            # Check ordering constraint if previous_score is provided
            if previous_score is not None and score > previous_score:
                return False, None, f"Score {score} is greater than previous row's score {previous_score}"

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

        First attempts to find an exact match (normalized for spaces and case).
        If no exact match is found, uses fuzzy string matching via SequenceMatcher
        to find the closest matching player name. The match must exceed the fuzzy
        threshold to be considered valid.

        Normalization converts names to lowercase and removes all spaces before comparison.

        Args:
            value: Extracted text for player name
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0.0-1.0).
                            Default 0.8 requires 80% similarity. Higher values are stricter.

        Returns:
            Tuple of (is_valid, matched_name, error_message):
                - is_valid (bool): Whether the name passed validation
                - matched_name (str | None): The canonical player name if valid, None otherwise
                - error_message (str): Descriptive error message if invalid, empty string if valid

        Example:
            >>> validator = CellValidator({'Alice Smith', 'Bob Jones'})
            >>> is_valid, name, error = validator.validate_player_name('alice smith')
            >>> print(is_valid, name)
            True Alice Smith
            >>> is_valid, name, error = validator.validate_player_name('Alce Smth', fuzzy_threshold=0.7)
            >>> print(is_valid, name)
            True Alice Smith
        """
        if not value:
            return False, None, "Player name is empty"

        # Normalize function to reuse
        def normalize(s: str) -> str:
            return value.lower().replace(" ", "")

        normalized_value = normalize(value)

        # Try exact match first (normalized)
        for name in self.valid_player_names:
            if normalized_value == normalize(name):
                return True, name, ""

        # Try fuzzy matching
        best_match = None
        best_score = 0

        for name in self.valid_player_names:
            # Calculate similarity ratio on normalized strings
            ratio = SequenceMatcher(None, normalized_value, normalize(name)).ratio()

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
        fuzzy_threshold: float,
        previous_place: Optional[int] = None,
        previous_score: Optional[int] = None
    ) -> Tuple[bool, Optional[any], str]:
        """
        Validate a cell value based on its column.

        Dispatches to the appropriate validation method based on the column index.
        This is the main entry point for validating individual table cells.

        Args:
            column: Column index indicating the cell type:
                   - 1: Place/rank (validated with validate_place)
                   - 2: Player name (validated with validate_player_name)
                   - 4: Score (validated with validate_score)
            value: Extracted text to validate
            fuzzy_threshold: Fuzzy match threshold for player names (0.0-1.0)
            previous_place: Previous row's place value for ordering constraint (optional)
            previous_score: Previous row's score value for ordering constraint (optional)

        Returns:
            Tuple of (is_valid, parsed_value, error_message):
                - is_valid (bool): Whether the cell passed validation
                - parsed_value (int | str | None): The validated value (type depends on column), None if invalid
                - error_message (str): Descriptive error message if invalid, empty string if valid

        Raises:
            ValueError: If column index is not 1, 2, or 4

        Example:
            >>> validator = CellValidator({'Alice'})
            >>> is_valid, value, error = validator.validate_cell(1, '5', 0.8)
            >>> print(is_valid, value)
            True 5
            >>> is_valid, value, error = validator.validate_cell(2, 'Alice', 0.8)
            >>> print(is_valid, value)
            True Alice
        """
        if column == 1:
            return self.validate_place(value, previous_place)
        elif column == 2:
            return self.validate_player_name(value, fuzzy_threshold)
        elif column == 4:
            return self.validate_score(value, previous_score)
        else:
            raise ValueError(f"Invalid column index: {column}")
