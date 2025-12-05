# Room Scanner - AI-Powered Asset Detection

A Flask web application that uses **YOLO object detection** and **OpenAI GPT-4 Vision** to automatically detect and classify items in room photos.

## Features

- 🔍 **Object Detection**: Uses YOLOe (YOLO Efficient) for fast, accurate object detection
- 🤖 **AI Classification**: OpenAI GPT-4 Vision API identifies and estimates value of detected items
- 📊 **Visual Results**: Displays bounding boxes and cropped images of detected assets
- 💰 **Value Estimation**: Provides estimated replacement values for detected items
- 🎯 **Smart Deduplication**: Removes duplicate detections using IoU thresholds and semantic grouping
- 📱 **Responsive UI**: Clean, modern interface with drag-and-drop image upload

## Architecture

```
Room Photo → YOLO Detection → Object Cropping → OpenAI Classification → Deduplication → Results
```

## Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster processing)

## Installation

### 1. Clone or navigate to the project directory

```bash
cd AI_Detect2
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first run will download the YOLO model (~50MB) automatically.

### 4. Set up environment variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-actual-api-key-here
```

Or export directly:

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

## Usage

### Start the Flask server

```bash
python app.py
```

The application will be available at: **http://localhost:5000**

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Click the upload box or drag and drop a room photo
3. Click "Scan Room" to process the image
4. View detected items with:
   - Item names (classified by AI)
   - Estimated replacement values
   - Confidence scores
   - Cropped images of each detected item
   - Visualization with bounding boxes

### Supported Image Formats

- JPG/JPEG
- PNG
- HEIC/HEIF (with additional setup, see below)

### Optional: HEIC/HEIF Support

To enable HEIC/HEIF image support (common for iPhone photos):

```bash
pip install pillow-heif
```

## Project Structure

```
AI_Detect2/
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── templates/
│   └── index.html             # Web UI
├── asset_detection/           # Detection module
│   ├── service.py            # Main service entry point
│   ├── detector.py           # YOLO object detection
│   ├── classifier.py         # OpenAI classification
│   ├── image_processor.py    # Image manipulation
│   ├── deduplicator.py       # Duplicate removal
│   ├── result_builder.py     # Result formatting
│   ├── file_handler.py       # File operations
│   ├── config.py             # Configuration settings
│   └── ...
├── uploads/                   # Temporary upload storage
├── outputs/
│   ├── crops/                # Cropped object images
│   └── visualizations/       # Annotated images
└── models/                    # YOLO models (auto-downloaded)
```

## Configuration

Edit `asset_detection/config.py` to customize:

- **YOLO Model Settings**: Size, confidence threshold, IoU threshold
- **OpenAI Model**: Choose between gpt-4o-mini (fast/cheap) or gpt-4o (slower/expensive)
- **Crop Size**: Target size for cropped images
- **Deduplication**: IoU thresholds for singleton and non-singleton items
- **Semantic Grouping**: Define which items should be unique per room

### Example Configurations

```python
# Higher confidence = fewer false positives
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Default: 0.1

# Use larger model for better accuracy (at cost of speed)
YOLO_MODEL_SIZE = 'l'  # Options: 's', 'm', 'l'

# Use GPT-4 for better classification quality
OPENAI_MODEL = "gpt-4o"  # Default: "gpt-4o-mini"
```

## API Endpoints

### `GET /`
Main page with upload interface

### `POST /upload`
Process uploaded image
- **Input**: Multipart form data with image file
- **Output**: JSON with detected assets
```json
{
  "success": true,
  "detected_assets": [...],
  "total_count": 5,
  "total_value": 2500,
  "visualization_url": "/visualizations/..."
}
```

### `GET /crops/<filename>`
Serve cropped asset images

### `GET /visualizations/<filename>`
Serve annotated visualization images

### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "service_available": true,
  "openai_key_set": true
}
```

## Troubleshooting

### "Service Not Available" Warning

This means ML dependencies are not installed. Run:
```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY must be set" Error

Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-key-here'
```

### YOLO Model Download Issues

If the YOLO model fails to download automatically:
1. Check your internet connection
2. The model will be downloaded to the `models/` directory
3. Manually download from Hugging Face if needed

### Low Detection Accuracy

Try adjusting in `asset_detection/config.py`:
- Lower `YOLO_CONFIDENCE_THRESHOLD` to detect more objects
- Use a larger YOLO model (`YOLO_MODEL_SIZE = 'l'`)
- Switch to GPT-4 for better classification

### Out of Memory Errors

- Close other applications
- Use a smaller YOLO model (`YOLO_MODEL_SIZE = 's'`)
- Reduce `CROP_TARGET_SIZE` in config
- Process smaller images

## Performance

### Typical Processing Times

- **YOLO Detection**: 1-3 seconds (CPU), <1 second (GPU)
- **OpenAI Classification**: 2-5 seconds per object (concurrent)
- **Total**: ~5-15 seconds for a typical room with 5-10 items

### Cost Estimation (OpenAI API)

Using `gpt-4o-mini`:
- ~$0.01 per image with 10 objects
- ~$1 for 100 room scans

Using `gpt-4o`:
- ~$0.05 per image with 10 objects
- ~$5 for 100 room scans

## Use Cases

- 📦 **Moving & Storage**: Inventory tracking before moves
- 🏠 **Property Management**: Asset management across properties
- 📝 **Insurance**: Documentation for claims
- 🏘️ **Real Estate**: Property listing generation
- 🧹 **Home Organization**: Room inventory tracking

## Technology Stack

- **Backend**: Flask 3.0
- **Computer Vision**: YOLOe v11 (Ultralytics)
- **AI Classification**: OpenAI GPT-4 Vision API
- **Image Processing**: OpenCV, Pillow
- **Async Processing**: aiohttp, asyncio
- **Deep Learning**: PyTorch

## Development

### Run in Debug Mode

```bash
# Already enabled in app.py
python app.py
```

### Run in Production

Use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

This project uses various open-source components. Please review their individual licenses.

## Credits

- YOLO Object Detection: [Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenAI GPT-4 Vision API: [OpenAI](https://openai.com/)
