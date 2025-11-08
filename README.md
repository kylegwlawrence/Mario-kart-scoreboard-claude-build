# Mario Kart Scoreboard OCR Pipeline

A Python-based optical character recognition (OCR) pipeline designed to extract race results from Mario Kart Nintendo Switch post-race scoreboard images. The system automatically detects the scoreboard table, extracts player names and scores, and validates results against expected constraints.

## Features

- **Multi-Engine OCR Support**: Choose between paddleocr, tesseract, or easyocr
  - easyocr is, unbelievably, the easiest and best OCR engine for this task so far. Therefor I only implement easyocr in my project right now
- **Configurable Preprocessing**: 14 different image preprocessing methods with chainable pipelines
- **Grid-Based Cell Extraction**: Extracts cells using configurable grid bounds for accurate alignment
- **Intelligent Validation**:
  - Exact matching for place (1-12)
  - Fuzzy matching for player names
  - Range validation for scores (1-999)
- **Retry Logic**: Per-cell retry with configurable strategies and preprocessing chains
- **Comprehensive Logging**: Standard and debug logging modes
- **Flexible Configuration**: JSON-based configuration files in `src/configs/` directory
- **Annotated Output**: Original images with gridlines, predictions, and confidence scores

## Installation

### Prerequisites
- Python 3.7+
- A Unix-like system (Linux, macOS, or WSL on Windows)

### Setup

1. **Clone/setup the repository**:
```bash
cd mario_kart_scores_claude_build
```

2. **Install dependencies**:
```bash
# Basic installation with required dependencies
pip install -r requirements.txt

# Or install specific OCR engines as needed
pip install easyocr                  # For easyocr (recommended)
pip install paddlepaddle paddleocr  # For paddleocr
pip install pytesseract tesseract    # For tesseract
```

3. **Verify installation**:
```bash
python3 -c "from src.orchestrator import OCRProcessor; print('Installation successful')"
```

## Usage

### Command Line Interface

#### Process a single image:
```bash
python main.py --image pngs/IMG_7995.png
```

#### Process all images in a directory:
```bash
python main.py --image-dir pngs/
```

#### Use a custom configuration:
```bash
python main.py --image pngs/IMG_7995.png --config src/configs/custom_config.json
```

#### Enable debug logging:
```bash
python main.py --image pngs/IMG_7995.png --debug
```

#### Get help:
```bash
python main.py --help
```

### Programmatic Usage

```python
from src.orchestrator import OCRProcessor

# Initialize processor
processor = OCRProcessor('src/configs/pipelines/default.json', debug=False)

# Process a single image
results = processor.process_image('pngs/IMG_7995.png')

# Access results
print(f"Valid predictions: {results['valid_predictions']}/{results['total_cells']}")
print(f"Failed cells: {results['failed_cells']}")
```

## Configuration

### Default Configuration
The default configuration is located in `src/configs/pipelines/default.json`. Additional configuration files include:
- `src/configs/grid.json`: Defines row and column cell bounds for the scoreboard
- `src/configs/ocr_engines.json`: Defines available OCR engines and character names CSV path

Create new JSON files in the `src/configs/pipelines/` directory to define custom pipeline configurations.

### Configuration Structure

```json
{
  "image_source": "./pngs",
  "output_paths": {
    "annotated": "output/annotated_images",
    "predictions": "output/predictions",
    "cell_images": "output/cell_images",
    "logs": ".logging"
  },
  "preprocessing_chains": [
    {
      "retry_attempt": 0,
      "methods": [
        {"method": "grayscale", "parameters": null},
        {"method": "gaussian_blur", "parameters": {"kernel": [5, 5], "sigmaX": 0, "sigmaY": 0}}
      ]
    }
  ],
  "primary_engine": "paddleocr",
  "retry_attempts": 3,
  "fuzzy_threshold": 0.8
}
```

### Available Preprocessing Methods

14 preprocessing methods are available for configurable image transformation pipelines:

#### Quick Reference Table

| Method | Required Params | Optional Params | Notes |
|--------|---|---|---|
| grayscale | - | - | No parameters |
| gaussian_blur | - | kernel, sigmaX, sigmaY | - |
| edge_detection | - | hysteresis_min, hysteresis_max | Canny algorithm |
| dilate | - | kernel, iterations | - |
| erode | - | kernel, iterations | - |
| threshold | - | threshold, max_value | Binary thresholding |
| adaptive_threshold | - | max_value, block_size, C | block_size auto-made odd |
| inversion | - | - | No parameters |
| morphology | - | operation, kernel | operation: open/close/gradient/tophat/blackhat |
| blur | - | kernel | Simple blur |
| contrast | - | alpha, beta | alpha: contrast, beta: brightness |
| median_blur | - | ksize | ksize auto-made odd |
| bilateral_filter | - | d, sigmaColor, sigmaSpace | Edge-preserving |
| downscale | scale_factor OR (width AND height) | - | Use one method |

#### Detailed Parameters Reference

**1. grayscale**
- No parameters needed
- Example: `{"method": "grayscale"}`

**2. gaussian_blur**
- `kernel` (tuple): Size of Gaussian kernel matrix for convolution; larger values produce more blur (must be positive odd numbers)
- `sigmaX` (float): Standard deviation in X direction; when 0, auto-calculated from kernel size; larger values increase horizontal blur spread
- `sigmaY` (float): Standard deviation in Y direction; when 0, equals sigmaX; controls vertical blur spread independently
- Example: `{"method": "gaussian_blur", "parameters": {"kernel": [5, 5], "sigmaX": 0, "sigmaY": 0}}`

**3. edge_detection**
- `hysteresis_min` (int): Lower threshold for Canny edge detector; pixels with gradient values below this are discarded as non-edges
- `hysteresis_max` (int): Upper threshold for Canny edge detector; pixels above this are strong edges; narrower gap between min/max produces more edge pixels
- Example: `{"method": "edge_detection", "parameters": {"hysteresis_min": 100, "hysteresis_max": 200}}`

**4. dilate**
- `kernel` (tuple): Size of rectangular structuring element; larger kernels expand bright regions more aggressively, thickening features
- `iterations` (int): Number of times dilation is applied; more iterations progressively expand bright regions, closing gaps and connecting nearby features
- Example: `{"method": "dilate", "parameters": {"kernel": [3, 3], "iterations": 1}}`

**5. erode**
- `kernel` (tuple): Size of rectangular structuring element; larger kernels shrink bright regions more aggressively, thinning features
- `iterations` (int): Number of times erosion is applied; more iterations progressively shrink bright regions, removing noise and separating connected features
- Example: `{"method": "erode", "parameters": {"kernel": [3, 3], "iterations": 1}}`

**6. threshold**
- `threshold` (int): Pixel intensity cutoff point (0-255); pixels above become max_value, pixels at or below become 0
- `max_value` (int): Value assigned to pixels exceeding threshold; becomes the "white" level in binary output
- Example: `{"method": "threshold", "parameters": {"threshold": 127, "max_value": 255}}`

**7. adaptive_threshold**
- `max_value` (int): Value assigned to pixels passing the adaptive threshold test; becomes the "white" level in binary output
- `block_size` (int): Size of pixel neighborhood for local threshold calculation (must be odd); larger values smooth threshold variations across the image
- `C` (float): Constant subtracted from weighted mean during thresholding; positive values make thresholding more aggressive (more black), negative more permissive
- Example: `{"method": "adaptive_threshold", "parameters": {"max_value": 255, "block_size": 11, "C": 2}}`

**8. inversion**
- No parameters needed
- Example: `{"method": "inversion"}`

**9. morphology**
- `operation` (string): Morphological operation type: 'open' (remove noise), 'close' (fill gaps), 'gradient' (detect edges), 'tophat' (extract bright features), 'blackhat' (extract dark features)
- `kernel` (tuple): Size of rectangular structuring element; larger kernels affect larger-scale structures in the image
- Example: `{"method": "morphology", "parameters": {"operation": "open", "kernel": [5, 5]}}`

**10. blur**
- `kernel` (tuple): Size of averaging kernel; each output pixel is the average of all pixels in this neighborhood; larger values produce stronger blur
- Example: `{"method": "blur", "parameters": {"kernel": [5, 5]}}`

**11. contrast**
- `alpha` (float): Contrast multiplier applied to pixel values; values > 1 increase contrast, < 1 decrease contrast (formula: output = alpha × input + beta)
- `beta` (int): Brightness offset added to all pixels; positive values brighten, negative darken; applied after alpha scaling
- Example: `{"method": "contrast", "parameters": {"alpha": 1.5, "beta": 10}}`

**12. median_blur**
- `ksize` (int): Aperture size for median filter (must be odd); each pixel becomes the median of its ksize × ksize neighborhood, removing salt-and-pepper noise while preserving edges
- Example: `{"method": "median_blur", "parameters": {"ksize": 5}}`

**13. bilateral_filter**
- `d` (int): Diameter of pixel neighborhood used during filtering; larger values consider more distant pixels but increase computation time
- `sigmaColor` (float): Standard deviation in color space; larger values mix pixels with larger intensity differences, reducing edge-preservation
- `sigmaSpace` (float): Standard deviation in coordinate space; larger values allow farther pixels to influence each other, increasing spatial smoothing
- Example: `{"method": "bilateral_filter", "parameters": {"d": 9, "sigmaColor": 75, "sigmaSpace": 75}}`

**14. downscale**
- **Option A - Scale factor (proportional):**
  - `scale_factor` (float): Proportional scaling factor; values < 1 shrink the image, > 1 enlarge it (e.g., 0.5 halves dimensions)
  - Example: `{"method": "downscale", "parameters": {"scale_factor": 0.5}}`
- **Option B - Explicit dimensions:**
  - `width` (int): Target image width in pixels; required if scale_factor not provided; must be specified with height
  - `height` (int): Target image height in pixels; required if scale_factor not provided; must be specified with width
  - Example: `{"method": "downscale", "parameters": {"width": 1920, "height": 1080}}`

## Output

### Directory Structure
```
output/
├── annotated_images/              # Annotated original images (JPG)
├── predictions/                   # CSV results
│   └── {image}_predictions.csv
├── cell_images/                   # Extracted cell images for debugging
└── logs/                          # Processing logs
.logging/                           # Debug logs
```

### CSV Output (`{image}_predictions.csv`)

Columns:
- **unique_key**: Unique identifier for the cell
- **row_id**: Row in table (0-11)
- **column_id**: Column in table (1, 2, or 4, where 1=place, 2=name, 4=score)
- **predicted_text**: Extracted and validated text
- **confidence**: OCR confidence score (0-1)
- **passes_validation**: Whether the cell passed validation constraints
- **text_coordinates**: Pixel coordinates in original image
- **original_filepath**: Path to original image
- **process_start_time**: Processing start time (ISO 8601)
- **process_end_time**: Processing completion time (ISO 8601)
- **primary_engine**: Primary OCR engine used
- **retry_attempt_used**: Which retry attempt succeeded (0 for first attempt)
- **pipeline_steps**: Preprocessing steps applied
- **pipeline_config_path**: Path to the configuration file used
- **failed_reason**: Reason for failure (if validation failed)
- **cell_image_paths**: Paths to extracted cell images

## Validation Rules

### Place (Column 1)
- Must be an integer in range 1-12
- Invalid values trigger retry

### Player Name (Column 2)
- Exact match: Returns the name as-is
- Fuzzy match: Uses fuzzy matching (configurable threshold, default 0.8) if exact match fails
- Valid names from the configured character CSV file

### Score (Column 4)
- Must be an integer in range 1-999
- Invalid values trigger retry

## Error Handling

- **Per-cell retry**: Failed cells trigger retry with different preprocessing/OCR engine
- **Configurable retry**: Customize max attempts via `retry_attempts` config parameter
- **Stop on critical error**: Batch processing stops on first critical/uncaught error
- **Graceful cell failure**: Individual cell validation failures are retried, not fatal
- **Comprehensive logging**: All errors and retries logged with full context

## Logging

### Standard Mode
Logs major events:
- Program start/completion
- File operations
- Processing summary
- Errors and warnings

### Debug Mode (--debug)
Verbose logging includes:
- Detailed transformation steps
- Fuzzy match scores
- Per-cell validation results
- Configuration details

Logs saved to `.logging/` directory with timestamps.

## Architecture

### Core Modules

1. **config_manager.py**: Configuration loading and validation
2. **preprocessing.py**: Image preprocessing pipeline
3. **ocr_engines.py**: Multi-engine OCR wrapper
4. **annotator.py**: Image annotation with grid lines and predictions
5. **constraint_validator.py**: Cell validation with constraints
6. **orchestrator.py**: Main orchestration module

### Processing Pipeline

```
Input Image
    ↓
[Preprocessing] → Grayscale, filters, thresholds, etc.
    ↓
[Cell Extraction] → Extract cells using configured grid bounds
    ↓
[OCR + Validation] → Extract text and validate against constraints
    ├→ Valid? → Store in predictions
    └→ Invalid? → Retry with different preprocessing chain (if attempts remaining)
    ↓
[Annotation] → Draw gridlines, OCR results, and confidence scores
    ↓
[Save Results] → CSV predictions, annotated image, cell images
```

## Performance Considerations

- **Batch Processing**: Process multiple images sequentially
- **Preprocessing Optimization**: Choose efficient preprocessing chains
- **OCR Selection**: paddleocr is fastest, easyocr is most accurate
- **Confidence Threshold**: Higher threshold = fewer retries, faster processing

## Troubleshooting

### No module named 'paddleocr'
```bash
pip install paddlepaddle paddleocr
```

### Results not as expected
- Check if image is clear and well-lit
- Try different preprocessing chain in config
- Verify grid bounds in `src/configs/grid.json` match your scoreboard layout

### OCR results inaccurate
- Adjust confidence threshold (lower = more lenient)
- Try different OCR engine
- Experiment with preprocessing methods

### Out of memory
- Process images in batches (default config does this)
- Reduce image resolution in preprocessing

## Development

### Code Formatting
```bash
black src/ main.py
```

### Type Checking
```bash
mypy src/ main.py
```

## Future Enhancements

- [ ] Multi-image batch processing optimization
- [ ] Result caching to avoid re-processing
- [ ] Web API interface
- [ ] Real-time video stream processing
- [ ] Custom model training for improved accuracy
- [ ] Distributed processing for large batches

## License

Specify your license here.

## Contributing

Contributions welcome! Please follow the code style (black, flake8).

## Support

For issues and questions, refer to the logging output and debug mode for detailed information about what's happening at each step.
