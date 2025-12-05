"""
Flask Room Scanner Application

A simple Flask web app for detecting and classifying assets/items in room images
using YOLO object detection and OpenAI GPT-4 Vision API.
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path

from asset_detection import AssetDetectionService

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CROPS_FOLDER'] = 'outputs/crops'
app.config['VIZ_FOLDER'] = 'outputs/visualizations'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic', 'heif'}

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['CROPS_FOLDER'], app.config['VIZ_FOLDER']]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Initialize detection service
try:
    detection_service = AssetDetectionService()
    service_available = detection_service._ml_available
except Exception as e:
    print(f"Failed to initialize detection service: {e}")
    service_available = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html', service_available=service_available)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and process for asset detection"""

    if not service_available:
        return jsonify({
            'success': False,
            'error': 'Asset detection service is not available. Please install ML dependencies.'
        }), 503

    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Check if filename is empty
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"\n{'='*60}")
        print(f"Processing uploaded file: {filename}")
        print(f"{'='*60}\n")

        # Process the image
        result = detection_service.process_image(
            filepath,
            save_crops=True,
            crop_storage_path=app.config['CROPS_FOLDER'],
            iou_threshold=None  # Use default from config
        )

        detected_assets = result.get('detected_assets', [])
        visualization_url = result.get('visualization_url')

        # Clean up uploaded file
        try:
            os.unlink(filepath)
        except:
            pass

        # Process results for response
        assets_for_response = []
        for asset in detected_assets:
            # crop_url is already a path string from result_builder
            crop_path = asset.get('crop_url')
            crop_url_formatted = f"/crops/{Path(crop_path).name}" if crop_path else None

            assets_for_response.append({
                'name': asset['name'],
                'confidence': asset['confidence'],
                'estimated_value': asset.get('estimated_value', 100),
                'bbox': asset['bbox'],
                'crop_url': crop_url_formatted
            })

        # Get visualization URL if available
        # visualization_url is a dict with 'path' and 'url' keys
        viz_url = None
        if visualization_url:
            viz_path = visualization_url.get('path') if isinstance(visualization_url, dict) else visualization_url
            viz_url = f"/visualizations/{Path(viz_path).name}"

        print(f"\n{'='*60}")
        print(f"✓ Processing complete! Found {len(assets_for_response)} items")
        print(f"{'='*60}\n")

        return jsonify({
            'success': True,
            'detected_assets': assets_for_response,
            'total_count': len(assets_for_response),
            'total_value': sum(a['estimated_value'] for a in assets_for_response),
            'visualization_url': viz_url
        })

    except Exception as e:
        print(f"\n❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        }), 500


@app.route('/crops/<filename>')
def serve_crop(filename):
    """Serve cropped asset images"""
    return send_from_directory(app.config['CROPS_FOLDER'], filename)


@app.route('/visualizations/<filename>')
def serve_visualization(filename):
    """Serve visualization images with bounding boxes"""
    return send_from_directory(app.config['VIZ_FOLDER'], filename)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service_available': service_available,
        'openai_key_set': bool(os.environ.get('OPENAI_API_KEY'))
    })


if __name__ == '__main__':
    # Check for required environment variables
    if not os.environ.get('OPENAI_API_KEY'):
        print("\n" + "="*60)
        print("⚠️  WARNING: OPENAI_API_KEY environment variable not set!")
        print("   Asset classification will not work.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("="*60 + "\n")

    print("\n" + "="*60)
    print("🚀 Room Scanner Flask App Starting...")
    print("="*60)
    print(f"Service Available: {service_available}")
    print(f"OpenAI Key Set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
