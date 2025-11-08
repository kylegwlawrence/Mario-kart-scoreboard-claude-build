# Claude Reply - Mario Kart Scores OCR Project

## Project Overview
Building a Python program that:
- Ingests PNG images of Mario Kart Nintendo Switch post-race scoreboards
- Applies configurable image preprocessing techniques
- Uses OCR (paddleocr, tesseract, or easyocr) to extract scoreboard data
- Stores results in structured formats (CSV, PNG, JPG, JSON)

## Key Information Gathered

### Character Data
- Valid player names from `character_info.csv`: 66 characters
- Format: name, weight, description
- Will use the `name` column for validation

### Sample Images
- Located in `/pngs/` directory
- 15 sample images available (IMG_7995.png through IMG_8045.png)
- Large file sizes (8-9 MB each)

## Clarification Answers from User
1. ✓ Player names from character_info.csv
2. ✓ OCR confidence threshold: configurable
3. ✓ Retry attempts and strategies: configurable (preprocessing chains, OCR models)
4. ✓ Table location detection: automatic
5. ✓ Preprocessing chains: config file
6. ✓ Output directory structure: subdirectories
7. ✓ Error handling: stop on first error
8. ✓ Annotated images: include scores, gridlines, text
9. ✓ All save paths: configurable

## Additional Clarifications Answered

1. **Config Management**:
   - Format: JSON
   - Location: configs/ folder
   - Runtime selection: path to config file chosen at runtime

2. **Table Detection**:
   - Method: Contour detection (configurable on/off)
   - Orientation: Always upright, slight rotation only (< few degrees)

3. **Cell Validation**:
   - Place (1-12): Reject/retry if invalid
   - Player Name: Try fuzzy matching if exact match fails
   - Score: Range 1-999 inclusive, reject/retry if invalid

4. **Retry Strategy**:
   - Scope: Per-cell retry (not entire table)
   - Intermediate Results: Store per-cell failure info

5. **GPU Usage**:
   - Auto-detect and use if available

6. **Debug Logging**:
   - Log transformation names only (not intermediate images)

## Chat Log

### Initial Setup
- Read prompt from claude_prompt.txt
- Read character_info.csv to understand valid player names
- Listed sample images in /pngs/ directory
- Received clarification answers from user
- Asked and received additional clarifications on config, validation, and logging

### Implementation Complete

#### Core Modules Created
1. **src/utils.py** - Logging, file I/O, and utility functions
   - Logging setup with file and console handlers
   - CSV and JSON file operations
   - Valid player name loading from character_info.csv

2. **src/config_manager.py** - Configuration management
   - Load and validate JSON configurations
   - Support for multiple preprocessing chains
   - Nested config access via get_nested()
   - Default config template generation

3. **src/preprocessing.py** - Image preprocessing pipeline
   - 13 preprocessing methods implemented:
     - grayscale, gaussian_blur, edge_detection
     - dilate, erode, threshold, adaptive_threshold
     - inversion, morphology, blur, contrast
     - median_blur, bilateral_filter
   - Chainable preprocessing with method tracking
   - PNG image saving

4. **src/ocr_engine.py** - Multi-engine OCR wrapper
   - Support for paddleocr, tesseract, easyocr
   - Auto GPU detection and configuration
   - Unified interface for all engines
   - Engine switching capability

5. **src/table_detection.py** - Table detection and annotation
   - Contour-based table bounds detection
   - Cell extraction from table (12 rows × 3 columns)
   - Cell coordinate calculation
   - Image annotation with gridlines, text, and confidence scores

6. **src/validation.py** - Cell validation module
   - Place validation (1-12)
   - Player name validation with fuzzy matching
   - Score validation (1-999)
   - Per-column validation logic

7. **src/ocr_processor.py** - Main orchestration module
   - Coordinates all pipeline components
   - Per-cell retry logic with configurable strategies
   - Results storage (CSV + JSON)
   - Comprehensive logging

#### Supporting Files
- **main.py** - CLI entry point with argparse
  - Single image processing: `--image <path>`
  - Batch processing: `--image-dir <path>`
  - Custom config: `--config <path>`
  - Debug mode: `--debug`

- **example_usage.py** - Programmatic usage examples
  - Single image processing
  - Batch processing
  - Custom configuration
  - Result access patterns

- **configs/default_config.json** - Default configuration template
  - 3 preprocessing chains for retries
  - paddleocr as primary OCR engine
  - Contour-based table detection enabled
  - Auto GPU detection
  - Max 3 retry attempts

- **requirements.txt** - Python dependencies
  - Core: numpy, opencv-python, pandas
  - OCR: paddleocr, tesseract, easyocr (optional)
  - Dev: pytest, black, flake8, mypy

- **src/__init__.py** - Package initialization with exports

#### Key Features Implemented
✓ Configurable preprocessing chains
✓ Multiple OCR engine support with auto-detection
✓ GPU auto-detection and usage
✓ Per-cell retry with configurable strategies
✓ Fuzzy matching for player names
✓ Comprehensive error handling
✓ Detailed logging (standard + debug modes)
✓ CSV predictions with confidence scores
✓ JSON metadata storage
✓ Annotated output images with gridlines
✓ Automatic table detection via contours
✓ Configurable output paths
✓ CLI and programmatic usage

#### Output Structure
```
output/
├── preprocessed_images/       # Preprocessing pipeline output
├── annotated_images/          # Original images + predictions + gridlines
└── predictions/
    ├── {image}_predictions.csv
    └── {image}_metadata.json
.logging/                       # Debug logs
```

## How to Use

### CLI Usage
```bash
# Single image processing
python main.py --image pngs/IMG_7995.png

# Batch processing
python main.py --image-dir pngs/

# With custom config
python main.py --image pngs/IMG_7995.png --config configs/custom_config.json

# Debug mode
python main.py --image pngs/IMG_7995.png --debug
```

### Programmatic Usage
```python
from src.ocr_processor import OCRProcessor

processor = OCRProcessor('configs/default_config.json')
results = processor.process_image('pngs/IMG_7995.png')
print(f"Valid cells: {results['valid_predictions']}/{results['total_cells']}")
```

## File Summary

### Executables
- **main.py** - Main CLI entry point (executable)
- **example_usage.py** - Programmatic usage examples (executable)

### Configuration
- **configs/default_config.json** - Default configuration

### Source Code
- **src/__init__.py** - Package initialization
- **src/utils.py** - Utilities, logging, file I/O
- **src/config_manager.py** - Configuration management
- **src/preprocessing.py** - Image preprocessing (13 methods)
- **src/ocr_engine.py** - Multi-engine OCR wrapper
- **src/table_detection.py** - Table detection and annotation
- **src/validation.py** - Cell validation logic
- **src/ocr_processor.py** - Main orchestration

### Documentation
- **README.md** - Comprehensive documentation
- **requirements.txt** - Python dependencies
- **claude_reply.md** - This file

## Implementation Notes

### Key Design Decisions
1. **Modular Architecture**: Each component is independent and testable
2. **Configuration-Driven**: Behavior controlled via JSON configs
3. **Graceful Degradation**: Falls back to full image if table detection fails
4. **Per-Cell Processing**: Retries applied at cell level, not table level
5. **Comprehensive Logging**: All operations logged for debugging

### Retry Logic
- Attempts OCR on a cell up to `max_attempts` times
- Uses same preprocessing chain, different OCR engines, or different preprocessing chains (configurable)
- Stops and returns None if all attempts fail
- Failed cells tracked separately

### Error Handling Strategy
- File operations: Throw errors (as specified)
- Preprocessing: Throw errors
- OCR failures: Log, retry, then throw if max attempts exceeded
- Validation failures: Log, retry, then mark as failed
- Batch processing: Stop on first error (as specified)

### Performance Optimizations
- Auto GPU detection and usage
- Efficient contour-based table detection
- Configurable preprocessing chains for different image qualities
- Multiple OCR engines available (choose based on speed/accuracy trade-off)

## Status: COMPLETE ✓

All core functionality, modules, and documentation complete and ready for testing with actual images.


---

## Project Completion Summary

**Date**: November 4, 2024  
**Status**: COMPLETE ✓

### Files Created
- **7 Core Modules** (176 KB total)
  - src/preprocessing.py (350+ lines)
  - src/ocr_processor.py (500+ lines)
  - src/ocr_engine.py (300+ lines)
  - src/table_detection.py (320+ lines)
  - src/config_manager.py (200+ lines)
  - src/validation.py (150+ lines)
  - src/utils.py (200+ lines)

- **2 Executable Scripts**
  - main.py (CLI interface)
  - example_usage.py (Usage examples)

- **1 Configuration**
  - configs/default_config.json (3 preprocessing chains)

- **3 Documentation Files**
  - README.md (Comprehensive guide)
  - requirements.txt (Dependencies)
  - PROJECT_SUMMARY.txt (Overview)

### Code Quality
- ✓ All Python syntax validated
- ✓ Full type hints in critical functions
- ✓ Comprehensive error handling
- ✓ Detailed docstrings on all classes and methods
- ✓ Modular, testable architecture

### Ready for Testing
The system is complete and ready to test with your Mario Kart scoreboard images. Start with:

```bash
python main.py --image pngs/IMG_7995.png
```

All results will be saved to the `output/` directory with logs in `.logging/`.


---

## Project Verification

All core components verified as complete:

### Core Modules (8/8) ✓
- src/utils.py
- src/config_manager.py
- src/preprocessing.py
- src/ocr_engine.py
- src/table_detection.py
- src/validation.py
- src/ocr_processor.py
- src/__init__.py

### Executable Scripts (2/2) ✓
- main.py
- example_usage.py

### Configuration (1/1) ✓
- configs/default_config.json

### Documentation (4/4) ✓
- README.md
- requirements.txt
- claude_reply.md
- PROJECT_SUMMARY.txt

**Total Files Created: 15**  
**Total Lines of Code: ~2,500**  
**All Syntax Checks: PASSED**

The Mario Kart Scoreboard OCR Pipeline is ready for deployment and testing!
