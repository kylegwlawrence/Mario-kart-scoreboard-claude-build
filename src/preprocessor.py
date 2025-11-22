"""
Image preprocessing module for OCR pipeline.
Handles various image transformations before OCR.
"""

import logging
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


class PreprocessingPipeline:
    """
    Image preprocessing pipeline for OCR preparation.

    Provides a modular system to apply multiple preprocessing transformations to images
    before OCR processing. Supports 14 different preprocessing methods including filtering,
    thresholding, morphological operations, and scaling. Methods are applied sequentially
    in the order specified by the configuration.

    Attributes:
        METHOD_HANDLERS: Dictionary mapping preprocessing method names to their handler functions
    """

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

    # Preprocessing methods
    @staticmethod
    def apply_grayscale(image: np.ndarray, parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Convert color image to grayscale.

        Args:
            image: Input image as numpy array
            parameters: Unused for grayscale conversion

        Returns:
            Grayscale image as numpy array. Returns unchanged if already grayscale.
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise and detail.

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'kernel': Tuple of (width, height) for blur kernel, defaults to (5, 5)
                - 'sigmaX': Gaussian kernel standard deviation in X direction, defaults to 0
                - 'sigmaY': Gaussian kernel standard deviation in Y direction, defaults to 0

        Returns:
            Blurred image as numpy array
        """
        kernel = tuple(parameters.get('kernel', (5, 5)))
        sigmaX = parameters.get('sigmaX', 0)
        sigmaY = parameters.get('sigmaY', 0)
        return cv2.GaussianBlur(image, kernel, sigmaX, sigmaY)

    @staticmethod
    def apply_edge_detection(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply Canny edge detection algorithm.

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'hysteresis_min': Lower threshold for edge detection, defaults to 100
                - 'hysteresis_max': Upper threshold for edge detection, defaults to 200

        Returns:
            Edge-detected image as numpy array
        """
        hysteresis_min = parameters.get('hysteresis_min', 100)
        hysteresis_max = parameters.get('hysteresis_max', 200)
        return cv2.Canny(image, hysteresis_min, hysteresis_max)

    @staticmethod
    def apply_dilate(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply morphological dilation to expand white regions.

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'kernel': Tuple of (width, height) for structuring element, defaults to (3, 3)
                - 'iterations': Number of dilation iterations, defaults to 1

        Returns:
            Dilated image as numpy array
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            tuple(parameters.get('kernel', (3, 3)))
        )
        iterations = parameters.get('iterations', 1)
        return cv2.dilate(image, kernel, iterations=iterations)

    @staticmethod
    def apply_erode(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply morphological erosion to shrink white regions.

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'kernel': Tuple of (width, height) for structuring element, defaults to (3, 3)
                - 'iterations': Number of erosion iterations, defaults to 1

        Returns:
            Eroded image as numpy array
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            tuple(parameters.get('kernel', (3, 3)))
        )
        iterations = parameters.get('iterations', 1)
        return cv2.erode(image, kernel, iterations=iterations)

    @staticmethod
    def apply_threshold(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply binary threshold to convert image to black and white.

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'threshold': Threshold value, pixels below become black, defaults to 127
                - 'max_value': Maximum value for white pixels, defaults to 255

        Returns:
            Binary thresholded image as numpy array
        """
        threshold = parameters.get('threshold', 127)
        max_value = parameters.get('max_value', 255)
        _, result = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
        return result

    @staticmethod
    def apply_adaptive_threshold(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply adaptive threshold with local neighborhood analysis.

        Uses Gaussian-weighted sum of neighborhood to determine threshold for each pixel.
        Useful for images with varying lighting conditions.

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'max_value': Maximum value for white pixels, defaults to 255
                - 'block_size': Size of neighborhood area (odd number), defaults to 11
                - 'C': Constant subtracted from weighted mean, defaults to 2

        Returns:
            Adaptively thresholded image as numpy array
        """
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
    def apply_inversion(image: np.ndarray, parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Invert all pixel values (black becomes white, white becomes black).

        Args:
            image: Input image as numpy array
            parameters: Unused for inversion

        Returns:
            Inverted image as numpy array
        """
        return cv2.bitwise_not(image)

    @staticmethod
    def apply_morphology(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply advanced morphological operations (open, close, gradient, etc.).

        Morphological operations are useful for removing noise, filling holes, or
        extracting shape features.

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'operation': Type of morphological operation, defaults to 'open'
                  Options: 'open', 'close', 'gradient', 'tophat', 'blackhat'
                - 'kernel': Tuple of (width, height) for structuring element, defaults to (5, 5)

        Returns:
            Morphologically processed image as numpy array
        """
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
        """
        Apply simple box blur filter.

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'kernel': Tuple of (width, height) for blur kernel, defaults to (5, 5)

        Returns:
            Blurred image as numpy array
        """
        kernel = tuple(parameters.get('kernel', (5, 5)))
        return cv2.blur(image, kernel)

    @staticmethod
    def apply_contrast(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Adjust image contrast (brightness multiplier) and brightness (additive offset).

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'alpha': Contrast multiplier (> 1.0 increases contrast), defaults to 1.0
                - 'beta': Brightness offset (positive brightens image), defaults to 0

        Returns:
            Contrast-adjusted image as numpy array
        """
        alpha = parameters.get('alpha', 1.0)
        beta = parameters.get('beta', 0)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def apply_median_blur(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply median blur filter (effective for salt-and-pepper noise removal).

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'ksize': Kernel size (odd number), defaults to 5

        Returns:
            Median-filtered image as numpy array
        """
        ksize = parameters.get('ksize', 5)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def apply_bilateral_filter(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Apply bilateral filter (edge-preserving blur).

        Blurs the image while preserving edges by considering both spatial and
        intensity differences.

        Args:
            image: Input image as numpy array
            parameters: Dict with optional keys:
                - 'd': Diameter of pixel neighborhood, defaults to 9
                - 'sigmaColor': Filter sigma in the color space, defaults to 75
                - 'sigmaSpace': Filter sigma in the coordinate space, defaults to 75

        Returns:
            Bilateral-filtered image as numpy array
        """
        d = parameters.get('d', 9)
        sigmaColor = parameters.get('sigmaColor', 75)
        sigmaSpace = parameters.get('sigmaSpace', 75)
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    @staticmethod
    def apply_downscale(image: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Downscale image using high-quality LANCZOS resampling.

        Supports either explicit target dimensions or proportional scaling via scale_factor.
        Works with both grayscale and color images.

        Args:
            image: Input image as numpy array
            parameters: Dict with required keys (choose one option):
                Option A - Explicit dimensions:
                - 'width': Target width in pixels
                - 'height': Target height in pixels
                Option B - Proportional scaling:
                - 'scale_factor': Scaling factor (0.5 = 50% of original size)

        Returns:
            Downscaled image as numpy array

        Raises:
            ValueError: If neither scale_factor nor both width and height are provided
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

        # Convert numpy array directly to PIL Image (PIL handles both grayscale and color)
        pil_image = Image.fromarray(image)

        # Resize using LANCZOS filter
        resized_image = pil_image.resize((width, height), Image.LANCZOS)

        # Convert back to numpy array
        return np.array(resized_image)
