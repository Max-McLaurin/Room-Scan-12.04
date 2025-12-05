import cv2
import base64
import numpy as np


class ImageProcessor:
    """Utility class for image processing operations"""

    @staticmethod
    def to_base64(image):
        """Convert OpenCV image to base64 for API"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def to_bytes(image):
        """Convert OpenCV image to bytes"""
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()

    @staticmethod
    def clean_filename(text):
        """Clean text for use in filename"""
        return text.lower().replace(' ', '_').replace('.', '').replace(',', '').replace(':', '')

    @staticmethod
    def create_visualization(image_path, detections, classifications):
        """
        Create visualization with bounding boxes and labels

        Args:
            image_path: Path to original image
            detections: List of detection objects with bbox info
            classifications: List of OpenAI classifications

        Returns:
            Bytes of the visualization image
        """
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Create a copy for drawing
        vis_image = image.copy()

        # Color palette for bounding boxes (BGR format)
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        # Draw each detection
        for i, (det, classification) in enumerate(zip(detections, classifications)):
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox

            # Select color (cycle through palette)
            color = colors[i % len(colors)]

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

            # Extract asset name from classification (handle dict or string)
            if isinstance(classification, dict):
                asset_name = classification.get('name', 'Unknown')
            else:
                asset_name = str(classification)

            # Prepare label
            label = f"{i+1}. {asset_name}"
            confidence = det['confidence']
            label_with_conf = f"{label} ({confidence:.0%})"

            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label_with_conf, font, font_scale, thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_height - baseline - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                vis_image,
                label_with_conf,
                (x1 + 5, y1 - baseline - 5),
                font,
                font_scale,
                (255, 255, 255),  # White text
                thickness,
                cv2.LINE_AA
            )

        # Add summary text at the top
        summary_text = f"Detected {len(detections)} assets"
        cv2.putText(
            vis_image,
            summary_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', vis_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
