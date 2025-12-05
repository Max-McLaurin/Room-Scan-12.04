"""
YOLO-World detector wrapper for open-vocabulary object detection.
"""

import torch
from pathlib import Path
import logging
from ultralytics import YOLOWorld
import cv2
import numpy as np
from PIL import Image

from .household_taxonomy import HouseholdTaxonomy
from . import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOWorldDetector:
    """
    YOLO-World detector with household item taxonomy.
    Provides open-vocabulary detection for 300+ household items.
    """

    def __init__(
        self,
        model_size: str = 'm',
        confidence_threshold: float = 0.20,
        iou_threshold: float = 0.5,
        use_custom_taxonomy: bool = True,
        room_type: str = None
    ):
        """
        Initialize YOLO-World detector.

        Args:
            model_size: Model size ('s', 'm', 'l', 'x')
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            use_custom_taxonomy: Use household taxonomy (True) or COCO classes (False)
            room_type: Optional room type for specialized detection ('kitchen', 'bedroom', etc.)
        """
        self.model_size = model_size.lower()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_custom_taxonomy = use_custom_taxonomy
        self.room_type = room_type

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = None
        self.taxonomy = HouseholdTaxonomy()
        self._load_model()

    def _load_model(self):
        """Load YOLO-World model from Ultralytics."""
        model_name = f'yolov8{self.model_size}-world.pt'

        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / model_name

        logger.info(f"Loading YOLO-World model: {model_name}")

        try:
            # Initialize YOLO-World model
            self.model = YOLOWorld(str(model_path))
            self.model.to(self.device)

            # Set custom classes from household taxonomy
            if self.use_custom_taxonomy:
                if self.room_type:
                    classes = self.taxonomy.get_room_specific_items(self.room_type)
                    logger.info(f"Loaded {len(classes)} {self.room_type}-specific items")
                else:
                    classes = self.taxonomy.get_all_items()
                    logger.info(f"Loaded {len(classes)} household items")

                self.model.set_classes(classes)

            logger.info(f"✓ YOLO-World model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            raise

    def detect_image(self, image_path):
        """
        Detect objects in an image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary containing detection results
        """
        try:
            # Load image
            image = self._load_image(image_path)

            # Run inference
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            # Process results
            detections = self._process_results(results[0])

            logger.info(f"YOLO-World detected {len(detections)} objects")

            return {
                'detections': detections,
                'image_size': image.size if isinstance(image, Image.Image) else image.shape[:2],
                'model_info': {
                    'model': f'yolov8{self.model_size}-world',
                    'confidence_threshold': self.confidence_threshold,
                    'iou_threshold': self.iou_threshold,
                    'num_classes': len(self.model.names)
                }
            }

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise

    def _load_image(self, image_path):
        """Load image from file path."""
        try:
            # Try OpenCV first
            image = cv2.imread(image_path)
            if image is None:
                # Fallback to PIL for HEIC/HEIF
                image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Could not load image: {image_path}, error: {e}")
            raise

    def _process_results(self, result):
        """Process YOLO results into standardized format."""
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                class_name = self.model.names[int(cls_id)]

                detection = {
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(conf),
                    'class_name': class_name,
                    'class_id': int(cls_id)
                }

                detections.append(detection)

        return detections

    def set_room_type(self, room_type):
        """
        Change room type and reload taxonomy.

        Args:
            room_type: Room type ('kitchen', 'bedroom', 'living_room', etc.)
        """
        self.room_type = room_type

        # Reload model with room-specific classes
        if self.use_custom_taxonomy:
            classes = self.taxonomy.get_room_specific_items(room_type)
            self.model.set_classes(classes)
            logger.info(f"Updated to {len(classes)} {room_type}-specific items")

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'model': f'yolov8{self.model_size}-world',
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'num_classes': len(self.model.names),
            'room_type': self.room_type,
            'taxonomy_enabled': self.use_custom_taxonomy
        }


# Example usage:
if __name__ == "__main__":
    detector = YOLOWorldDetector(model_size='m', confidence_threshold=0.20)
    results = detector.detect_image('test_image.jpg')

    print(f"Detected {len(results['detections'])} objects:")
    for det in results['detections']:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
