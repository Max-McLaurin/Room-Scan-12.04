# Quick Start Guide - Room Scanner

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Installation

### 1. Set up Python environment

```bash
# Navigate to project directory
cd AI_Detect2

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download YOLO model (~50MB) automatically.

### 3. Set up OpenAI API key

```bash
# Set environment variable
export OPENAI_API_KEY='your-openai-api-key-here'

# OR create .env file
cp .env.example .env
# Then edit .env and add your key
```

### 4. Run the app

```bash
python app.py
```

Open your browser to: **http://localhost:5000**

## Usage

1. **Upload** a room photo (click or drag-and-drop)
2. **Click** "Scan Room"
3. **View** detected items with:
   - Item names
   - Estimated values
   - Confidence scores
   - Cropped images
   - Bounding box visualization

## Troubleshooting

### "Service Not Available"
```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='your-key-here'
```

### Port already in use
```bash
# Change port in app.py:
app.run(debug=True, host='0.0.0.0', port=5001)
```

## What's Next?

- See [README.md](README.md) for full documentation
- See [implementation_start.md](implementation_start.md) for technical details
- Adjust settings in `asset_detection/config.py`

## Cost Estimation

Using `gpt-4o-mini`:
- ~$0.01 per room with 10 items
- ~$1 for 100 scans

## Support

For issues or questions:
1. Check [README.md](README.md) for detailed troubleshooting
2. Review configuration in `asset_detection/config.py`
3. Check logs in terminal for error messages

---

**Ready to scan your first room!** 🏠✨
