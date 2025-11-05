# Mario Kart Scoreboard OCR Pipeline

A Python-based optical character recognition (OCR) pipeline designed to extract race results from Mario Kart Nintendo Switch post-race scoreboard images. The system automatically detects the scoreboard table, extracts player names and scores, and validates results against expected constraints.

## Features

- **Multi-Engine OCR Support**: Choose between paddleocr, tesseract, or easyocr
- **Configurable Preprocessing**: 13 different image preprocessing methods with chainable pipelines
- **Automatic Table Detection**: Uses contour detection to locate the scoreboard
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
pip install paddlepaddle paddleocr  # For paddleocr (recommended)
pip install pytesseract tesseract    # For tesseract
pip install easyocr                  # For easyocr
```

3. **Verify installation**:
```bash
python3 -c "from src.ocr_processor import OCRProcessor; print('Installation successful')"
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
from src.ocr_processor import OCRProcessor

# Initialize processor
processor = OCRProcessor('src/configs/default_config.json', debug=False)

# Process a single image
results = processor.process_image('pngs/IMG_7995.png')

# Access results
print(f"Valid predictions: {results['valid_predictions']}/{results['total_cells']}")
print(f"Failed cells: {results['failed_cells']}")
```

See `example_usage.py` for more detailed examples.

## Configuration

### Default Configuration
The default configuration is located in `src/configs/default_config.json`. Create new JSON files in the `src/configs/` directory to define custom configurations.

### Configuration Structure

```json
{
  "image_source": "./pngs",
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
        {"method": "grayscale", "parameters": null},
        {"method": "gaussian_blur", "parameters": {"kernel": [5, 5], "sigmaX": 0, "sigmaY": 0}}
      ]
    }
  ],
  "ocr_config": {
    "engines": ["paddleocr", "tesseract", "easyocr"],
    "primary_engine": "paddleocr",
    "confidence_threshold": 0.5
  },
  "table_detection": {
    "enabled": true,
    "method": "contour"
  },
  "retry_config": {
    "max_attempts": 3,
    "retry_on_low_confidence": true,
    "retry_strategies": ["preprocessing", "ocr_engine"]
  },
  "character_names_csv": "character_info.csv"
}
```

### Available Preprocessing Methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| `grayscale` | None | Convert to grayscale |
| `gaussian_blur` | kernel, sigmaX, sigmaY | Apply Gaussian blur |
| `edge_detection` | hysteresis_min, hysteresis_max | Canny edge detection |
| `dilate` | kernel, iterations | Dilate morphology |
| `erode` | kernel, iterations | Erode morphology |
| `threshold` | threshold, max_value | Binary threshold |
| `adaptive_threshold` | max_value, block_size, C | Adaptive threshold |
| `inversion` | None | Invert colors |
| `morphology` | operation, kernel | Morphological operations |
| `blur` | kernel | Simple blur |
| `contrast` | alpha, beta | Adjust contrast |
| `median_blur` | ksize | Median blur |
| `bilateral_filter` | d, sigmaColor, sigmaSpace | Bilateral filter |

## Output

### Directory Structure
```
output/
├── preprocessed_images/           # Preprocessed images (PNG)
├── annotated_images/              # Annotated original images (JPG)
└── predictions/                   # CSV and JSON results
    ├── {image}_predictions.csv
    └── {image}_metadata.json
.logging/                           # Debug logs
```

### CSV Output (`{image}_predictions.csv`)

Columns:
- **row_id**: Row in table (0-11)
- **column_id**: Column in table (0-2, where 0=place, 1=name, 2=score)
- **predicted_text**: Extracted and validated text
- **confidence**: OCR confidence score (0-1)
- **text_coordinates**: Pixel coordinates in original image
- **original_filepath**: Path to original image
- **preprocessed_filepath**: Path to preprocessed image
- **process_start_timestamp**: Processing start time (ISO 8601)
- **preprocessing_methods**: Preprocessing steps applied

### JSON Output (`{image}_metadata.json`)

Contains:
- Image paths and processing timestamps
- Configuration used (preprocessing, OCR, retry settings)
- Summary of results (valid/failed cells)
- Details of failed cells with error messages

## Validation Rules

### Place (Column 0)
- Must be an integer in range 1-12
- Invalid values trigger retry

### Player Name (Column 1)
- Exact match: Returns the name as-is
- Fuzzy match: Uses fuzzy matching (threshold 0.8) if exact match fails
- Valid names from `character_info.csv`

### Score (Column 2)
- Must be an integer in range 1-999
- Invalid values trigger retry

## Error Handling

- **Graceful degradation**: If table detection fails, uses full image bounds
- **Per-cell retry**: Failed cells trigger retry with different preprocessing/OCR engine
- **Configurable retry**: Customize max attempts and retry strategies
- **Stop on first error**: Batch processing stops on first critical error
- **Comprehensive logging**: All errors logged with full context

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
3. **ocr_engine.py**: Multi-engine OCR wrapper
4. **table_detection.py**: Table bounds detection and annotation
5. **validation.py**: Cell validation with constraints
6. **ocr_processor.py**: Main orchestration module

### Processing Pipeline

```
Input Image
    ↓
[Preprocessing] → Grayscale, filters, thresholds, etc.
    ↓
[Saved] → output/preprocessed_images/
    ↓
[Table Detection] → Find table bounds via contours
    ↓
[Cell Extraction] → Extract 12×3 cells from table
    ↓
[OCR + Validation] → Extract text and validate
    ├→ Valid? → Store in predictions
    └→ Invalid? → Retry (if attempts remaining)
    ↓
[Annotation] → Draw gridlines, text, confidence
    ↓
[Save Results] → CSV, JSON, annotated image
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

### Table not detected correctly
- Check if image is clear and well-lit
- Try different preprocessing chain in config
- Disable table detection: `"enabled": false` (will use full image)

### OCR results inaccurate
- Adjust confidence threshold (lower = more lenient)
- Try different OCR engine
- Experiment with preprocessing methods

### Out of memory
- Process images in batches (default config does this)
- Reduce image resolution in preprocessing

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ main.py example_usage.py
```

### Type Checking
```bash
mypy src/ main.py example_usage.py
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
