"""
YOLOE Detector Wrapper

A comprehensive wrapper for the YOLOE model providing easy-to-use detection
with text prompts, visual prompts, and prompt-free inference.
"""

import torch
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOEDetector:
    """
    YOLOE model wrapper for object detection and segmentation.
    
    Supports three modes:
    - Text prompts: Natural language descriptions
    - Visual prompts: Visual examples for detection  
    - Prompt-free: Built-in vocabulary detection
    """
    
    AVAILABLE_MODELS = {
        # Text/Visual prompt models
        'yoloe-v8-s': 'yoloe-v8s-seg.pt',
        'yoloe-v8-m': 'yoloe-v8m-seg.pt', 
        'yoloe-v8-l': 'yoloe-v8l-seg.pt',
        'yoloe-11-s': 'yoloe-11s-seg.pt',
        'yoloe-11-m': 'yoloe-11m-seg.pt',
        'yoloe-11-l': 'yoloe-11l-seg.pt',
        # Prompt-free models
        'yoloe-v8-s-pf': 'yoloe-v8s-seg-pf.pt',
        'yoloe-v8-m-pf': 'yoloe-v8m-seg-pf.pt', 
        'yoloe-v8-l-pf': 'yoloe-v8l-seg-pf.pt',
        'yoloe-11-s-pf': 'yoloe-11s-seg-pf.pt',
        'yoloe-11-m-pf': 'yoloe-11m-seg-pf.pt',
        'yoloe-11-l-pf': 'yoloe-11l-seg-pf.pt',
        # COCO transfer models
        'yoloe-v8-s-coco': 'yoloe-v8s-seg-coco.pt',
        'yoloe-v8-m-coco': 'yoloe-v8m-seg-coco.pt', 
        'yoloe-v8-l-coco': 'yoloe-v8l-seg-coco.pt',
        'yoloe-11-s-coco': 'yoloe-11s-seg-coco.pt',
        'yoloe-11-m-coco': 'yoloe-11m-seg-coco.pt',
        'yoloe-11-l-coco': 'yoloe-11l-seg-coco.pt'
    }
    
    def __init__(
        self,
        model_size: str = 's',
        model_version: str = 'v8',
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5
    ):
        """
        Initialize YOLOE detector.

        Args:
            model_size: Model size ('s', 'm', 'l')
            model_version: Model version ('v8', '11')
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_size = model_size.lower()
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Model setup
        self.model = None
        self.model_path = None
        self._load_model()
        
    def _get_model_key(self, prompt_type: str = 'text') -> str:
        """Get the appropriate model key based on prompt type."""
        base_key = f'yoloe-{self.model_version}-{self.model_size}'
        
        if prompt_type == 'free':
            model_key = f'{base_key}-pf'
        elif prompt_type == 'coco':
            model_key = f'{base_key}-coco'
        else:  # text or visual
            model_key = base_key
            
        return model_key
    
    def _load_model(self, prompt_type: str = 'text'):
        """Load the YOLOE model from Hugging Face."""
        model_key = self._get_model_key(prompt_type)
        
        if model_key not in self.AVAILABLE_MODELS:
            available = ', '.join(self.AVAILABLE_MODELS.keys())
            raise ValueError(f"Model {model_key} not available. Choose from: {available}")
        
        model_filename = self.AVAILABLE_MODELS[model_key]
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / model_filename
        
        # Download model if not exists
        if not model_path.exists():
            logger.info(f"Downloading {model_filename}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id="jameslahm/yoloe",
                    filename=model_filename,
                    local_dir=models_dir,
                    local_dir_use_symlinks=False
                )
                self.model_path = Path(downloaded_path)
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise
        else:
            self.model_path = model_path
            
        logger.info(f"Model loaded from: {self.model_path}")
        
        # Load model with ultralytics
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    
    def detect_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: Optional[str] = None,
        prompt_type: str = 'text',
        return_masks: bool = True
    ) -> Dict:
        """
        Detect objects in an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            prompt: Text prompt for detection (e.g., "person, car, dog")
            prompt_type: Type of prompt ('text', 'visual', 'free')
            return_masks: Whether to return segmentation masks

        Returns:
            Dictionary containing detection results
        """
        # Load appropriate model based on prompt type
        current_model_key = self._get_model_key(prompt_type)
        expected_filename = self.AVAILABLE_MODELS[current_model_key]
        
        # Reload model if different prompt type
        if self.model is None or not str(self.model_path).endswith(expected_filename):
            logger.info(f"Loading model for prompt type: {prompt_type}")
            self._load_model(prompt_type)
        
        # Process input image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image format")
        
        # Run inference
        try:
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Process results
            detections = self._process_results(results[0], return_masks)
            
            # Create results
            detection_results = {
                'detections': detections,
                'image_size': image.size,
                'model_info': {
                    'model': f"yoloe-{self.model_version}-{self.model_size}",
                    'confidence_threshold': self.confidence_threshold,
                    'iou_threshold': self.iou_threshold
                }
            }

            return detection_results
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def _process_results(self, result, return_masks: bool = True) -> List[Dict]:
        """Process YOLO results into standardized format."""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            # Handle class names
            if hasattr(result, 'names') and result.names:
                class_names = [result.names.get(int(cls), f"class_{int(cls)}") 
                             for cls in result.boxes.cls.cpu().numpy()]
            else:
                class_names = [f"object_{i}" for i in range(len(boxes))]
            
            # Handle masks if available and requested
            masks = None
            if return_masks and hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
            
            for i, (box, conf, class_name) in enumerate(zip(boxes, confidences, class_names)):
                detection = {
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(conf),
                    'class_name': class_name,
                    'class_id': i
                }
                
                if masks is not None and i < len(masks):
                    detection['mask'] = masks[i]
                
                detections.append(detection)
        
        return detections

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            'model_key': f"yoloe-{self.model_version}-{self.model_size}",
            'model_path': str(self.model_path) if self.model_path else None,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold
        }

        return info