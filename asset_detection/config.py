"""
Configuration for Asset Detection Module

This module centralizes all configuration parameters for the asset detection system,
making it easy to tune and adjust settings without modifying multiple files.
"""


# ================================
# YOLO Detection Model Settings
# ================================

# YOLOe model configuration
YOLO_MODEL_SIZE = 'm'  # Options: 's' (small), 'm' (medium), 'l' (large)
YOLO_MODEL_VERSION = '11'  # Options: 'v8', '11'
YOLO_CONFIDENCE_THRESHOLD = 0.1  # Minimum confidence for object detection (0.0 - 1.0)
YOLO_IOU_THRESHOLD = 0.5  # IoU threshold for Non-Maximum Suppression (0.0 - 1.0)


# ================================
# Image Processing Settings
# ================================

# Target size for cropped asset images (pixels)
CROP_TARGET_SIZE = 224  # Standard size for consistency in ML pipelines


# ================================
# OpenAI Classification Settings
# ================================

# Model to use for asset classification
OPENAI_MODEL = "gpt-4o-mini"  # Options: "gpt-4o-mini", "gpt-4o", etc.

# Maximum number of assets to categorize in a single batch request
# Adjust based on token limits and asset complexity
MAX_BATCH_CATEGORIZATION_SIZE = 50


# ================================
# Duplicate Detection Settings
# ================================

# General IoU threshold for duplicate removal via NMS
DEFAULT_IOU_THRESHOLD = 0.5

# Confidence threshold for filtering low-confidence detections
MIN_CONFIDENCE_THRESHOLD = 0.25

# IoU thresholds for semantic duplicate removal
SINGLETON_IOU_THRESHOLD = 0.1  # For unique items (sink, faucet, etc.)
NON_SINGLETON_IOU_THRESHOLD = 0.6  # For items that can have multiples (cabinets, chairs, etc.)


# ================================
# Semantic Grouping Configuration
# ================================

# Items that should ALWAYS be forced to a single entry, regardless of IoU overlap
# This is for distributed items like cabinets where individual sections don't overlap
FORCE_SINGLE_ENTRY = [
    'cabinet', 'cabinets', 'kitchen cabinet', 'cabinet door',
    'cabinetry', 'upper cabinet', 'lower cabinet', 'base cabinet', 'wall cabinet'
]

# Room-specific singleton items
KITCHEN_SINGLETONS = [
    'sink', 'faucet', 'sponge', 'dishwasher', 'oven',
    'stove', 'refrigerator', 'microwave', 'toaster oven',
    'soap dispenser', 'spray bottle', 'knife block',
    'mat', 'air fryer', 'toaster', 'coffee maker',
    'cabinet', 'cabinets', 'kitchen cabinet', 'cabinet door'
]

BEDROOM_SINGLETONS = [
    'bed', 'dresser', 'wardrobe', 'closet', 'armoire',
    'vanity', 'desk', 'nightstand', 'night stand'
]

LIVING_ROOM_SINGLETONS = [
    'sofa', 'couch', 'tv', 'television', 'coffee table',
    'entertainment center', 'tv stand'
]

# Combined list of all singleton items
# These items should typically be unique in a room and will be aggressively deduplicated
SINGLETON_ITEM_TYPES = KITCHEN_SINGLETONS + BEDROOM_SINGLETONS + LIVING_ROOM_SINGLETONS

# Semantic grouping for similar items
# Items in the same group will be considered semantically equivalent
SEMANTIC_GROUPS = {
    'sink': ['sink', 'kitchen sink', 'double sink'],
    'faucet': ['faucet', 'tap', 'water faucet', 'kitchen faucet'],
    'sponge': ['sponge', 'cleaning sponge', 'dish sponge'],
    'dishwasher': ['dishwasher', 'dish washer'],
    'oven': ['oven', 'toaster oven', 'microwave oven'],
    'stove': ['stove', 'cooktop', 'range'],
    'refrigerator': ['refrigerator', 'fridge', 'freezer'],
    'cabinet': ['cabinet', 'cabinets', 'kitchen cabinet', 'cabinet door', 'upper cabinet', 'lower cabinet', 'base cabinet', 'wall cabinet', 'cabinetry'],
}


# ================================
# File Storage Settings
# ================================

# Directory for storing YOLO model files
MODEL_STORAGE_DIR = 'models'

# S3 storage paths (relative to bucket root)
CROP_STORAGE_BASE_PATH = 'assets/crops'
VISUALIZATION_STORAGE_BASE_PATH = 'assets/visualizations'
