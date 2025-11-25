"""
Classes and methods to track race progress through a series
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from src.orchestrator import OCRProcessor
from src.utils import load_csv


class Race:
    """
    Represents a single race within a series.

    Stores race data including OCR results, file paths, and player results.
    Provides methods to access and correct race data.

    Attributes:
        race_number: Race number in the series (1-indexed)
        image_path: Path to original scoreboard image
        run_id: Unique OCR processing run identifier
        predictions_csv_path: Path to raw predictions CSV file
        annotated_image_path: Path to annotated scoreboard image
        results: List of extracted player results with format:
                 [{"player": str, "character": str, "place": int, "score": int, "confidence": float}, ...]
        players: Reference to character name → player name mapping dictionary
    """

    def __init__(
        self,
        race_number: int,
        image_path: str,
        run_id: str,
        predictions_csv_path: str,
        annotated_image_path: str,
        results: List[Dict[str, Any]],
        players: Dict[str, str]
    ):
        """
        Initialize a Race instance.

        Args:
            race_number: Race number in the series (1-indexed)
            image_path: Path to original scoreboard image
            run_id: Unique OCR processing run identifier
            predictions_csv_path: Path to raw predictions CSV file
            annotated_image_path: Path to annotated scoreboard image
            results: List of extracted player results
            players: Dictionary mapping character names to player names

        Raises:
            ValueError: If race_number < 1
        """
        if race_number < 1:
            raise ValueError(f"race_number must be >= 1, got {race_number}")

        self.race_number = race_number
        self.image_path = image_path
        self.run_id = run_id
        self.predictions_csv_path = predictions_csv_path
        self.annotated_image_path = annotated_image_path
        self.results = results
        self.players = players

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get structured race results.

        Returns:
            List of dictionaries with format:
            [{"player": str, "character": str, "place": int, "score": int, "confidence": float}, ...]
        """
        return self.results

    def get_annotated_image(self) -> str:
        """
        Get path to annotated image.

        Returns:
            File path to annotated image as string
        """
        return self.annotated_image_path

    def get_predictions_data(self) -> str:
        """
        Get path to raw predictions CSV.

        Returns the file path as a string. The CSV contains all OCR attempts per cell
        with full metadata. Can be parsed by the user as needed.

        Returns:
            File path to predictions CSV as string
        """
        return self.predictions_csv_path

    def correct_result(
        self,
        character_name: str,
        place: Optional[int] = None,
        score: Optional[int] = None
    ) -> None:
        """
        Manually correct OCR results for a specific character.

        Updates the race results. Only fields that are provided (not None)
        will be updated.

        Args:
            character_name: Character name whose results need correction
            place: Corrected placement (1-12), or None to keep current value
            score: Corrected score (1-999), or None to keep current value

        Raises:
            ValueError: If character_name not found in players dictionary
        """
        if character_name not in self.players:
            raise ValueError(f"Character '{character_name}' not in players dictionary")

        # Find and update the result for this character
        found = False
        for result in self.results:
            if result['character'] == character_name:
                if place is not None:
                    result['place'] = place
                if score is not None:
                    result['score'] = score
                found = True
                break

        if not found:
            raise ValueError(f"Character '{character_name}' not found in race {self.race_number} results")

    def correct_results(self, corrections: Dict[str, Dict[str, int]]) -> None:
        """
        Apply multiple corrections in a single call.

        Args:
            corrections: Dictionary with format:
                {"Character Name": {"place": 3, "score": 850},
                 "Other Character": {"place": 1, "score": 999}}
                Fields within each character's dict are optional (place and/or score)

        Raises:
            ValueError: If any character not found in players dictionary
        """
        for character_name, correction_data in corrections.items():
            place = correction_data.get('place')
            score = correction_data.get('score')
            self.correct_result(character_name, place, score)


class RaceSeries:
    """
    Tracks race progress and results for all players across a series of races.

    Manages multiple Race objects representing individual races in a series.

    Attributes:
        players: Dictionary mapping character names to player names
        num_races: Total number of races in the series
        races: Dictionary storing Race objects indexed by race_number
        ocr_processor: OCRProcessor instance for processing scoreboard images
    """

    def __init__(self, players: dict, num_races: int, config_path: Optional[str] = None, debug: bool = False):
        """
        Initialize a RaceSeries instance.

        Args:
            players: Dictionary mapping character names to player names
                     (e.g., {"Shy Guy": "Kyle", "Peach": "Sarah"})
            num_races: Total number of races in the series
            config_path: Path to OCR pipeline configuration file. If None, uses default path.
            debug: Enable debug logging in OCRProcessor

        Raises:
            ValueError: If num_races < 1 or players dict is empty
            IOError: If config_path cannot be loaded
        """
        if num_races < 1:
            raise ValueError(f"num_races must be >= 1, got {num_races}")

        if not players:
            raise ValueError("players dictionary cannot be empty")

        self.players = players
        self.num_races = num_races
        self.races = {}  # {race_number: Race}

        # Initialize OCR processor with config
        if config_path is None:
            config_path = "src/configs/pipelines/default.json"

        try:
            self.ocr_processor = OCRProcessor(config_path, debug=debug)
        except Exception as e:
            raise IOError(f"Failed to initialize OCRProcessor with config {config_path}: {e}")

    def add_scoreboard_image(self, image_path: str, race_number: int) -> Dict[str, Any]:
        """
        Process a scoreboard image and store race results.

        Automatically runs OCR processing on the image and creates a Race object.

        Args:
            image_path: Path to the scoreboard image file
            race_number: Race number in the series (1-indexed, must be 1 to num_races)

        Returns:
            Dictionary with processing status containing:
                - 'success': bool - whether processing succeeded
                - 'run_id': str - unique identifier for this processing run
                - 'predictions_csv_path': str - path to raw predictions CSV
                - 'annotated_image_path': str - path to annotated image
                - 'race_results': list - extracted player results with format:
                    [{"player": str, "character": str, "place": int, "score": int}, ...]

        Raises:
            ValueError: If race_number is invalid or out of range
            IOError: If image cannot be processed
        """
        if race_number < 1 or race_number > self.num_races:
            raise ValueError(f"race_number must be between 1 and {self.num_races}, got {race_number}")

        try:
            # Process image with OCR
            ocr_results = self.ocr_processor.process_image(image_path)

            run_id = ocr_results.get('output_file', '').split('_')[-2]  # Extract run_id from filename
            predictions_csv_path = ocr_results.get('output_file', '')

            # Generate annotated image path
            image_stem = Path(image_path).stem
            annotated_image_path = str(
                Path(self.ocr_processor.output_paths['annotated']) /
                f"{image_stem}_{run_id}_annotated.jpg"
            )

            # Parse predictions CSV to extract best results per cell
            race_results = self._parse_predictions_csv(predictions_csv_path)

            # Create and store Race object
            race = Race(
                race_number=race_number,
                image_path=image_path,
                run_id=run_id,
                predictions_csv_path=predictions_csv_path,
                annotated_image_path=annotated_image_path,
                results=race_results,
                players=self.players
            )
            self.races[race_number] = race

            return {
                'success': True,
                'run_id': run_id,
                'predictions_csv_path': predictions_csv_path,
                'annotated_image_path': annotated_image_path,
                'race_results': race_results
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _parse_predictions_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """
        Parse predictions CSV and extract best results per cell.

        Maps character names to player names and extracts place/score/confidence data.

        Args:
            csv_path: Path to predictions CSV file

        Returns:
            List of dictionaries with extracted race results
        """
        try:
            csv_data = load_csv(csv_path)
        except IOError:
            return []

        # Group attempts by (row, col) and find best result per cell
        cells = {}  # (row, col) -> best_attempt_dict

        for row_data in csv_data:
            row_id = int(row_data.get('row_id', -1))
            col_id = int(row_data.get('column_id', -1))
            cell_key = (row_id, col_id)

            # Skip if not one of our target columns (1=place, 2=character, 4=score)
            if col_id not in [1, 2, 4]:
                continue

            # Only consider validated results
            if row_data.get('passes_validation', 'False') != 'True':
                continue

            # Keep best result (highest confidence)
            if cell_key not in cells:
                cells[cell_key] = row_data
            else:
                current_conf = float(cells[cell_key].get('confidence', 0))
                new_conf = float(row_data.get('confidence', 0))
                if new_conf > current_conf:
                    cells[cell_key] = row_data

        # Organize results by row (player) with character, place, score
        results_by_row = {}  # row -> {character, place, score, confidence}

        for (row, col), cell_data in cells.items():
            if row not in results_by_row:
                results_by_row[row] = {
                    'character': None,
                    'place': None,
                    'score': None,
                    'confidence': 0.0
                }

            validated_text = cell_data.get('validated_text', '')
            confidence = float(cell_data.get('confidence', 0))

            if col == 1:  # Place
                results_by_row[row]['place'] = int(validated_text) if validated_text else None
            elif col == 2:  # Character
                results_by_row[row]['character'] = validated_text
                results_by_row[row]['confidence'] = confidence
            elif col == 4:  # Score
                results_by_row[row]['score'] = int(validated_text) if validated_text else None

        # Format final results with player names
        race_results = []
        for row in sorted(results_by_row.keys()):
            row_data = results_by_row[row]
            character = row_data['character']

            # Map character to player name
            player = self.players.get(character, character)  # Fallback to character if not found

            race_results.append({
                'player': player,
                'character': character,
                'place': row_data['place'],
                'score': row_data['score'],
                'confidence': row_data['confidence']
            })

        return race_results

    def get_race_results(self, race_number: int) -> List[Dict[str, Any]]:
        """
        Get structured race results for a specific race.

        Args:
            race_number: Race number in the series (1-indexed)

        Returns:
            List of dictionaries with format:
            [{"player": str, "character": str, "place": int, "score": int, "confidence": float}, ...]

        Raises:
            ValueError: If race_number is invalid
            KeyError: If race has not been processed yet
        """
        if race_number < 1 or race_number > self.num_races:
            raise ValueError(f"race_number must be between 1 and {self.num_races}, got {race_number}")

        if race_number not in self.races:
            raise KeyError(f"Race {race_number} has not been processed yet")

        return self.races[race_number].get_results()

    def get_annotated_image(self, race_number: int) -> str:
        """
        Get path to annotated image for a specific race.

        Args:
            race_number: Race number in the series (1-indexed)

        Returns:
            File path to annotated image as string

        Raises:
            ValueError: If race_number is invalid
            KeyError: If race has not been processed yet
        """
        if race_number < 1 or race_number > self.num_races:
            raise ValueError(f"race_number must be between 1 and {self.num_races}, got {race_number}")

        if race_number not in self.races:
            raise KeyError(f"Race {race_number} has not been processed yet")

        return self.races[race_number].get_annotated_image()

    def get_predictions_data(self, race_number: int) -> str:
        """
        Get path to raw predictions CSV for a specific race.

        Returns the file path as a string. The CSV contains all OCR attempts per cell
        with full metadata. Can be parsed by the user as needed.

        Args:
            race_number: Race number in the series (1-indexed)

        Returns:
            File path to predictions CSV as string

        Raises:
            ValueError: If race_number is invalid
            KeyError: If race has not been processed yet
        """
        if race_number < 1 or race_number > self.num_races:
            raise ValueError(f"race_number must be between 1 and {self.num_races}, got {race_number}")

        if race_number not in self.races:
            raise KeyError(f"Race {race_number} has not been processed yet")

        return self.races[race_number].get_predictions_data()

    def correct_race_result(self, race_number: int, character_name: str,
                           place: Optional[int] = None, score: Optional[int] = None) -> None:
        """
        Manually correct OCR results for a specific character in a race.

        Updates the in-memory race results. Only fields that are provided (not None)
        will be updated.

        Args:
            race_number: Race number in the series (1-indexed)
            character_name: Character name whose results need correction
            place: Corrected placement (1-12), or None to keep current value
            score: Corrected score (1-999), or None to keep current value

        Raises:
            ValueError: If race_number is invalid or character_name not found
            KeyError: If race has not been processed yet
        """
        if race_number < 1 or race_number > self.num_races:
            raise ValueError(f"race_number must be between 1 and {self.num_races}, got {race_number}")

        if race_number not in self.races:
            raise KeyError(f"Race {race_number} has not been processed yet")

        self.races[race_number].correct_result(character_name, place, score)

    def correct_race_results(self, race_number: int, corrections: Dict[str, Dict[str, int]]) -> None:
        """
        Apply multiple corrections to a race in a single call.

        Args:
            race_number: Race number in the series (1-indexed)
            corrections: Dictionary with format:
                {"Character Name": {"place": 3, "score": 850},
                 "Other Character": {"place": 1, "score": 999}}
                Fields within each character's dict are optional (place and/or score)

        Raises:
            ValueError: If race_number is invalid or any character not found
            KeyError: If race has not been processed yet
        """
        if race_number < 1 or race_number > self.num_races:
            raise ValueError(f"race_number must be between 1 and {self.num_races}, got {race_number}")

        if race_number not in self.races:
            raise KeyError(f"Race {race_number} has not been processed yet")

        self.races[race_number].correct_results(corrections)

    def get_series_standings(self) -> Dict[str, Any]:
        """
        Get aggregate standings for the entire race series.

        Placeholder method for future implementation of series-level analytics.

        Returns:
            Dictionary with placeholder structure for series standings
        """
        return {
            'placeholder': True,
            'message': 'Series standings to be implemented'
        }
    