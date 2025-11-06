"""
OCR engine wrapper supporting multiple OCR libraries.
Handles paddleocr, tesseract, and easyocr.
"""

import logging
import numpy as np
from typing import Any, List, Optional, Tuple
from paddleocr import PaddleOCR
import pytesseract
import easyocr


class OCREngine:
    """Unified OCR interface for multiple engines."""

    SUPPORTED_ENGINES = ['paddleocr', 'tesseract', 'easyocr']

    def __init__(
        self,
        primary_engine: str = 'paddleocr',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize OCR engine.

        Args:
            primary_engine: Primary OCR engine to use
            logger: Logger instance

        Raises:
            ValueError: If engine is not supported
            ImportError: If required library is not installed
        """
        if primary_engine not in self.SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported engine: {primary_engine}. Must be one of {self.SUPPORTED_ENGINES}")

        self.primary_engine = primary_engine
        self.logger = logger

        if self.logger:
            self.logger.info(f"Initializing {primary_engine} OCR engine")

        # Initialize the engine
        self.engine = self._initialize_engine(primary_engine)

    def _initialize_engine(self, engine_name: str) -> Any:
        """
        Initialize the specified OCR engine.

        Args:
            engine_name: Name of the engine to initialize

        Returns:
            Initialized OCR engine object

        Raises:
            Exception: If engine initialization fails
        """
        try:
            if engine_name == 'paddleocr':
                return PaddleOCR(
                    use_angle_cls=True,
                    lang='en'
                )

            elif engine_name == 'tesseract':
                return pytesseract

            elif engine_name == 'easyocr':
                return easyocr.Reader(['en'])

        except Exception as e:
            raise Exception(f"Failed to initialize {engine_name}: {e}")

    def extract_text(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.0
    ) -> List[Tuple[str, float, List[Tuple[float, float]]]]:
        """
        Extract text from image using primary engine.

        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence score (0-1)

        Returns:
            List of tuples: (text, confidence, coordinates)
            where coordinates are list of (x, y) points

        Raises:
            Exception: If OCR fails
        """
        try:
            if self.primary_engine == 'paddleocr':
                return self._extract_text_paddleocr(image, confidence_threshold)

            elif self.primary_engine == 'tesseract':
                return self._extract_text_tesseract(image, confidence_threshold)

            elif self.primary_engine == 'easyocr':
                return self._extract_text_easyocr(image, confidence_threshold)

        except Exception as e:
            if self.logger:
                self.logger.error(f"OCR extraction failed: {e}")
            raise

    def _extract_text_paddleocr(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.0
    ) -> List[Tuple[str, float, List[Tuple[float, float]]]]:
        """Extract text using PaddleOCR."""
        try:
            result = self.engine.ocr(image, cls=True)

            if not result or not result[0]:
                return []

            extractions = []
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]

                if confidence >= confidence_threshold:
                    # Convert coordinates from list of lists to list of tuples
                    coords = [(float(p[0]), float(p[1])) for p in line[0]]
                    extractions.append((text, confidence, coords))

            return extractions

        except Exception as e:
            raise Exception(f"PaddleOCR extraction failed: {e}")

    def _extract_text_tesseract(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.0
    ) -> List[Tuple[str, float, List[Tuple[float, float]]]]:
        """Extract text using Tesseract."""
        try:
            # Get detailed data including confidence
            data = self.engine.image_to_data(image, output_type='dict')

            extractions = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue

                # Tesseract confidence is 0-100
                confidence = data['conf'][i] / 100.0

                if confidence >= confidence_threshold:
                    x, y = data['left'][i], data['top'][i]
                    w, h = data['width'][i], data['height'][i]
                    # Create bounding box coordinates
                    coords = [
                        (x, y),
                        (x + w, y),
                        (x + w, y + h),
                        (x, y + h)
                    ]
                    extractions.append((text, confidence, coords))

            return extractions

        except Exception as e:
            raise Exception(f"Tesseract extraction failed: {e}")

    def _extract_text_easyocr(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.0
    ) -> List[Tuple[str, float, List[Tuple[float, float]]]]:
        """Extract text using EasyOCR."""
        try:
            result = self.engine.readtext(image)

            if not result:
                return []

            extractions = []
            for detection in result:
                coords, text, confidence = detection

                if confidence >= confidence_threshold:
                    # Coordinates are already in the right format
                    coords_list = [(float(p[0]), float(p[1])) for p in coords]
                    extractions.append((text, confidence, coords_list))

            return extractions

        except Exception as e:
            raise Exception(f"EasyOCR extraction failed: {e}")