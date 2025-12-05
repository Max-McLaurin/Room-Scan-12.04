"""
Asset result builder

This module handles building final asset detection results including:
- Deduplication coordination
- S3 crop uploads
- Result formatting
- Progress logging
"""

import asyncio
from .deduplicator import AssetDeduplicator


class AssetResultBuilder:
    """
    Builds final asset detection results from raw detections, classifications, and categorizations.

    Responsibilities:
    - Coordinate deduplication process
    - Handle async S3 uploads for crop images
    - Format results into standardized output
    - Log progress and results
    """

    def __init__(self, image_processor, file_handler):
        """
        Initialize the result builder with required dependencies.

        Args:
            image_processor: ImageProcessor instance for image operations
            file_handler: FileHandler instance for S3 operations
        """
        self.deduplicator = AssetDeduplicator()
        self.image_processor = image_processor
        self.file_handler = file_handler

    async def build_results_async(self, cropped_objects, classifications, categorizations,
                                  save_crops=False, crop_storage_path=None):
        """
        Build final results from detected objects, classifications, and categorizations.

        This is the main entry point that orchestrates:
        1. Deduplication
        2. S3 uploads (if enabled)
        3. Result formatting

        Args:
            cropped_objects: List of detected object data (bbox, confidence, images, etc.)
            classifications: List of OpenAI classifications for each object
            categorizations: List of category assignments for each object
            save_crops: Whether to save crop images to S3
            crop_storage_path: S3 path for storing crop images

        Returns:
            List of formatted asset result dictionaries
        """
        # Remove semantic duplicates
        print(f"\nRemoving semantic duplicates from {len(cropped_objects)} detections...")
        cropped_objects, classifications = self.deduplicator.remove_duplicates(
            cropped_objects, classifications
        )

        # Filter categorizations to match deduplicated objects
        # Note: This assumes deduplication preserves order
        # Handle None categorizations (standalone mode without database)
        if categorizations is not None:
            categorizations = categorizations[:len(cropped_objects)]
        else:
            categorizations = [None] * len(cropped_objects)

        print(f"After deduplication: {len(cropped_objects)} unique assets\n")

        # Handle S3 uploads for crop images
        crop_urls = await self._handle_crop_uploads(
            cropped_objects, classifications, save_crops, crop_storage_path
        )

        # Build and return final results
        return self._build_result_list(
            cropped_objects, classifications, categorizations, crop_urls
        )

    async def _handle_crop_uploads(self, cropped_objects, classifications,
                                   save_crops, crop_storage_path):
        """
        Handle async S3 uploads for crop images.

        Args:
            cropped_objects: List of detected objects with images
            classifications: List of OpenAI classifications
            save_crops: Whether to upload crops
            crop_storage_path: S3 storage path

        Returns:
            List of crop data dicts (url, path) or None for each object
        """
        if save_crops and crop_storage_path:
            upload_tasks = self._prepare_crop_upload_tasks(
                cropped_objects, classifications, crop_storage_path
            )
            print(f"Uploading {len(upload_tasks)} crops to S3 concurrently...")
            return await asyncio.gather(*upload_tasks)
        else:
            return [None] * len(cropped_objects)

    def _prepare_crop_upload_tasks(self, cropped_objects, classifications, crop_storage_path):
        """
        Create async upload tasks for all detected crops.

        Args:
            cropped_objects: List of detected objects with images
            classifications: List of OpenAI classifications
            crop_storage_path: S3 storage path

        Returns:
            List of async tasks for uploading crops
        """
        upload_tasks = []
        for i, (obj, openai_classification) in enumerate(zip(cropped_objects, classifications)):
            asset_name, _ = self.extract_asset_info(openai_classification)
            clean_name = self.image_processor.clean_filename(asset_name)
            filename = f"{i+1}_{clean_name}.jpg"
            image_bytes = self.image_processor.to_bytes(obj['image'])
            task = self.file_handler.save_crop_async(image_bytes, crop_storage_path, filename)
            upload_tasks.append(task)
        return upload_tasks

    def _build_result_list(self, cropped_objects, classifications, categorizations, crop_urls):
        """
        Build the final list of formatted asset results.

        Args:
            cropped_objects: List of detected objects
            classifications: List of OpenAI classifications
            categorizations: List of category assignments
            crop_urls: List of crop data (url, path) or None

        Returns:
            List of formatted asset result dictionaries
        """
        detected_assets = []

        for i, (obj, classification, categorization, crop_data) in enumerate(
            zip(cropped_objects, classifications, categorizations, crop_urls)
        ):
            result = self._format_single_result(obj, classification, categorization, crop_data)
            detected_assets.append(result)
            self._log_result(i, obj, classification, categorization, result)

        return detected_assets

    def _format_single_result(self, obj, classification, categorization, crop_data):
        """
        Format a single asset detection result.

        Args:
            obj: Detected object data
            classification: OpenAI classification
            categorization: Category assignment
            crop_data: Crop image data (url, path) or None

        Returns:
            Formatted result dictionary
        """
        asset_name, estimated_value = self.extract_asset_info(classification)

        return {
            'detection_id': obj['detection_id'],
            'yolo_detection': str(obj['class_name']),
            'openai_classification': asset_name,
            'name': asset_name,  # Alias for scripts compatibility
            'estimated_value': int(estimated_value),
            'confidence': float(obj['confidence']),
            'bbox': [int(x) for x in obj['bbox']],
            'crop_image': crop_data['url'] if crop_data else None,
            'crop_image_path': crop_data['path'] if crop_data else None,
            'crop_url': crop_data['url'] if crop_data else None,  # Alias for app compatibility
            'category_id': categorization['category_id'] if categorization else None,
            'category_name': categorization['category_name'] if categorization else None,
            'parent_category_name': categorization.get('parent_category_name') if categorization else None,
        }

    def _log_result(self, index, obj, classification, categorization, result):
        """
        Log a formatted result for debugging/monitoring.

        Args:
            index: Result index (0-based)
            obj: Detected object data
            classification: OpenAI classification
            categorization: Category assignment
            result: Formatted result dictionary
        """
        asset_name, _ = self.extract_asset_info(classification)

        if categorization:
            category_info = f", Category={categorization.get('parent_category_name', '')} > {categorization['category_name']}"
        else:
            category_info = ""

        print(
            f"✓ RESULT {index+1} (ID: {obj['detection_id'][:8]}...): "
            f"YOLO={obj['class_name']}, OpenAI={asset_name} "
            f"(${result['estimated_value']}){category_info}"
        )

    @staticmethod
    def extract_asset_info(openai_classification):
        """
        Extract asset name and estimated value from OpenAI classification output.

        Supports both dict responses and string fallback.

        Args:
            openai_classification: OpenAI API response (dict or string)

        Returns:
            Tuple of (asset_name, estimated_value)
        """
        if isinstance(openai_classification, dict):
            asset_name = openai_classification.get('name', 'Unknown Item')
            estimated_value = openai_classification.get('estimated_value', 100)
        else:
            asset_name = str(openai_classification)
            estimated_value = 100

        return asset_name, int(estimated_value)
