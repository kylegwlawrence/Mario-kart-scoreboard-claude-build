"""
Image preprocessing module for OCR pipeline.
Handles various image transformations before OCR.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


class PreprocessingPipeline:
    """Applies a chain of preprocessing methods to an image."""

    # Mapping of method names to their handler functions
    METHOD_HANDLERS = {
        'grayscale': 'apply_grayscale',
        'gaussian_blur': 'apply_gaussian_blur',
        'edge_detection': 'apply_edge_detection',
        'dilate': 'apply_dilate',
        'erode': 'apply_erode',
        'threshold': 'apply_threshold',
        'adaptive_threshold': 'apply_adaptive_threshold',
        'inversion': 'apply_inversion',
        'morphology': 'apply_morphology',
        'blur': 'apply_blur',
        'contrast': 'apply_contrast',
        'median_blur': 'apply_median_blur',
        'bilateral_filter': 'apply_bilateral_filter',
        'downscale': 'apply_downscale',
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize preprocessing pipeline.

        Args:
            logger: Logger instance
        """
        self.logger = logger

    def apply_chain(
        self,
        image: np.ndarray,
        methods: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply a chain of preprocessing methods to an image.

        Args:
            image: Input image as numpy array
            methods: List of preprocessing method dicts with 'method' and 'parameters' keys

        Returns:
            Tuple of (processed_image, applied_methods_list)

        Raises:
            ValueError: If method is not recognized
            Exception: If preprocessing fails
        """
        result_image = image.copy()
        applied_methods = []

        try:
            for method_config in methods:
                method_name = method_config.get('method')
                parameters = method_config.get('parameters', {}) or {}

                if method_name not in self.METHOD_HANDLERS:
                    raise ValueError(f"Unknown preprocessing method: {method_name}")

                handler_name = self.METHOD_HANDLERS[method_name]
                handler = getattr(self, handler_name)

                if self.logger:
                    self.logger.debug(f"Applying preprocessing: {method_name}")

                result_image = handler(result_image, parameters)
                applied_methods.append(f"{method_name}({parameters})")

            if self.logger:
                self.logger.info(f"Applied preprocessing chain with {len(methods)} methods")

            return result_image, applied_methods

        except Exception as e:
            if self.logger:
                self.logger.error(f"Preprocessing failed: {e}")
            raise

    def save_preprocessed_image(
        self,
        image: np.ndarray,
        output_path: str
    ) -> None:
        """
        Save preprocessed image as PNG.

        Args:
            image: Preprocessed image
            output_path: Path to save PNG file

        Raises:
            IOError: If file writing fails
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(output_path, image)
            if not success:
                raise IOError(f"cv2.imwrite returned False for {output_path}")
            if self.logger:
                self.logger.info(f"Saved preprocessed image to {output_path}")
        except Exception as e:
            raise IOError(f"Failed to save preprocessed image to {output_path}: {e}")

    # Preprocessing methods
    @staticmethod
    def apply_grayscale(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply Gaussian blur."""
        kernel = tuple(parameters.get('kernel', (5, 5)))
        sigmaX = parameters.get('sigmaX', 0)
        sigmaY = parameters.get('sigmaY', 0)
        return cv2.GaussianBlur(image, kernel, sigmaX, sigmaY)

    @staticmethod
    def apply_edge_detection(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply edge detection using Canny algorithm."""
        hysteresis_min = parameters.get('hysteresis_min', 100)
        hysteresis_max = parameters.get('hysteresis_max', 200)
        return cv2.Canny(image, hysteresis_min, hysteresis_max)

    @staticmethod
    def apply_dilate(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply dilation."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            tuple(parameters.get('kernel', (3, 3)))
        )
        iterations = parameters.get('iterations', 1)
        return cv2.dilate(image, kernel, iterations=iterations)

    @staticmethod
    def apply_erode(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply erosion."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            tuple(parameters.get('kernel', (3, 3)))
        )
        iterations = parameters.get('iterations', 1)
        return cv2.erode(image, kernel, iterations=iterations)

    @staticmethod
    def apply_threshold(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply binary threshold."""
        threshold = parameters.get('threshold', 127)
        max_value = parameters.get('max_value', 255)
        _, result = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
        return result

    @staticmethod
    def apply_adaptive_threshold(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply adaptive threshold."""
        max_value = parameters.get('max_value', 255)
        block_size = parameters.get('block_size', 11)
        C = parameters.get('C', 2)
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            image,
            max_value,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )

    @staticmethod
    def apply_inversion(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Invert image colors."""
        return cv2.bitwise_not(image)

    @staticmethod
    def apply_morphology(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply morphological operations."""
        operation_name = parameters.get('operation', 'open')
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            tuple(parameters.get('kernel', (5, 5)))
        )

        operations = {
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE,
            'gradient': cv2.MORPH_GRADIENT,
            'tophat': cv2.MORPH_TOPHAT,
            'blackhat': cv2.MORPH_BLACKHAT,
        }

        op = operations.get(operation_name, cv2.MORPH_OPEN)
        return cv2.morphologyEx(image, op, kernel)

    @staticmethod
    def apply_blur(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply simple blur."""
        kernel = tuple(parameters.get('kernel', (5, 5)))
        return cv2.blur(image, kernel)

    @staticmethod
    def apply_contrast(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Adjust contrast and brightness."""
        alpha = parameters.get('alpha', 1.0)
        beta = parameters.get('beta', 0)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def apply_median_blur(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply median blur."""
        ksize = parameters.get('ksize', 5)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def apply_bilateral_filter(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply bilateral filter."""
        d = parameters.get('d', 9)
        sigmaColor = parameters.get('sigmaColor', 75)
        sigmaSpace = parameters.get('sigmaSpace', 75)
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    @staticmethod
    def apply_downscale(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Downscale image using PIL's high-quality LANCZOS resampling.

        Supports either explicit dimensions or scale_factor for proportional scaling.
        """
        width = parameters.get('width')
        height = parameters.get('height')
        scale_factor = parameters.get('scale_factor')

        # Determine target dimensions
        original_height, original_width = image.shape[:2]

        if scale_factor is not None:
            # Scale both dimensions proportionally
            width = int(original_width * scale_factor)
            height = int(original_height * scale_factor)
        elif width is None or height is None:
            raise ValueError("Downscale requires either 'scale_factor' or both 'width' and 'height' parameters")

        # Convert numpy array (BGR) to PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Resize using LANCZOS filter
        resized_image = pil_image.resize((width, height), Image.LANCZOS)

        # Convert back to numpy array (BGR)
        return cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
