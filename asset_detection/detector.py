import cv2
import uuid

from .yoloe_detector import YOLOEDetector
from .bbox_utils import calculate_square_crop_bounds, remove_duplicates_by_iou  # type: ignore
from . import config


class ObjectDetector:
    """Handles YOLO object detection and image cropping"""

    def __init__(self):
        # Load YOLOe detector (prompt-free model for better asset detection)
        self.model = YOLOEDetector(
            model_size=config.YOLO_MODEL_SIZE,
            model_version=config.YOLO_MODEL_VERSION,
            confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD,
            iou_threshold=config.YOLO_IOU_THRESHOLD
        )


    def _detect_objects(self, image_path):
        """Run YOLOe object detection and return raw detections"""
        results = self.model.detect_image(
            image_path,
            prompt_type='free',  # prompt-free mode for general object detection
            return_masks=False
        )

        all_detections = []
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            all_detections.append({
                'confidence': float(det['confidence']),
                'class_id': det.get('class_id', 0),
                'class_name': det['class_name'],
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            })

        print(f"YOLOe detected {len(all_detections)} objects")
        return all_detections

    def _crop_and_resize_detection(self, image, detection, target_size=None):
        """Crop a single detection as a square and resize it"""
        if target_size is None:
            target_size = config.CROP_TARGET_SIZE

        bbox = detection['bbox']
        crop_y1, crop_y2, crop_x1, crop_x2 = calculate_square_crop_bounds(bbox, image.shape)

        # Validate crop bounds
        if crop_y1 >= crop_y2 or crop_x1 >= crop_x2:
            print(f"Warning: Invalid crop bounds ({crop_y1}, {crop_y2}, {crop_x1}, {crop_x2}), skipping detection")
            return None

        # Crop the square region
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Validate cropped image is not empty
        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            print(f"Warning: Empty cropped image for detection, skipping")
            return None

        # Resize to standard size for consistency
        try:
            cropped_resized = cv2.resize(cropped, (target_size, target_size))
            return cropped_resized
        except cv2.error as e:
            print(f"Warning: Failed to resize cropped image ({cropped.shape}): {e}, skipping detection")
            return None

    def _crop_detections(self, image, detections):
        """Crop all detections from the image"""
        cropped_objects = []
        for detection in detections:
            cropped_image = self._crop_and_resize_detection(image, detection)
            
            # Skip invalid cropped images
            if cropped_image is None:
                print(f"Skipping invalid cropped image for detection: {detection.get('class_name', 'unknown')}")
                continue

            cropped_objects.append({
                'detection_id': str(uuid.uuid4()),
                'image': cropped_image,
                'confidence': detection['confidence'],
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'bbox': detection['bbox']
            })

        return cropped_objects

    def detect_and_crop(self, image_path, iou_threshold=None):
        """Detect objects and crop them as squares

        Note: confidence_threshold parameter is kept for API compatibility but
        YOLOe uses its own configured thresholds (set in __init__)
        """
        if iou_threshold is None:
            iou_threshold = config.DEFAULT_IOU_THRESHOLD
        # Try loading with OpenCV first
        image = cv2.imread(image_path)
        if image is None:
            # Fallback to PIL if OpenCV fails (handles HEIC/HEIF better)
            try:
                from PIL import Image
                import numpy as np
                pil_image = Image.open(image_path).convert('RGB')
                image = np.array(pil_image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                print(f"Loaded image using PIL fallback: {image_path}")
            except Exception as e:
                print(f"Could not load image: {image_path}, error: {e}")
                return []

        # Detect objects
        all_detections = self._detect_objects(image_path)

        # Remove duplicates based on IoU
        unique_detections = remove_duplicates_by_iou(all_detections, iou_threshold)
        print(f"Filtered {len(all_detections)} detections down to {len(unique_detections)} unique objects (IoU threshold: {iou_threshold})")

        # Crop and return the unique detections
        return self._crop_detections(image, unique_detections)
