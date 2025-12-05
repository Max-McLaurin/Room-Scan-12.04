import asyncio
from .result_builder import AssetResultBuilder
from .classification_orchestrator import ClassificationOrchestrator


class AssetDetectionService:
    """Django service for detecting assets in property room images"""

    def __init__(self):
        # Lazy import - only load ML dependencies when actually needed
        try:
            from .detector import ObjectDetector
            from .image_processor import ImageProcessor
            from .file_handler import FileHandler

            self.detector = ObjectDetector()
            self.image_processor = ImageProcessor()
            self.file_handler = FileHandler()

            # Initialize orchestrators with dependencies
            self.classification_orchestrator = ClassificationOrchestrator(
                self.image_processor
            )
            self.result_builder = AssetResultBuilder(
                self.image_processor,
                self.file_handler
            )

            self._ml_available = True
        except ImportError as e:
            self._ml_available = False
            self._ml_error = str(e)
            print(f"⚠️  Asset detection unavailable: {e}")
            print("   Install ML dependencies with: pip install -r requirements/ml.txt")

    async def process_image_async(self, image_path, save_crops=False, crop_storage_path=None, iou_threshold=None):
        """Main processing function with async OpenAI calls and S3 uploads"""
        if not self._ml_available:
            raise RuntimeError(
                f"Asset detection requires ML dependencies. {self._ml_error}\n"
                "Install with: pip install -r requirements/ml.txt"
            )

        print(f"Processing: {image_path}")

        # Detect and crop objects
        cropped_objects = self.detector.detect_and_crop(image_path, iou_threshold)

        if not cropped_objects:
            print("No objects detected")
            return {'detected_assets': [], 'visualization_url': None}

        # Classify and categorize all detected objects
        classifications, categorizations = await self.classification_orchestrator.classify_and_categorize(
            cropped_objects
        )

        # Process results with async S3 uploads
        detected_assets = await self.result_builder.build_results_async(
            cropped_objects, classifications, categorizations, save_crops, crop_storage_path
        )

        # Create and upload visualization
        visualization_url = None
        if save_crops and crop_storage_path:
            print("Creating visualization with bounding boxes...")
            viz_bytes = self.image_processor.create_visualization(image_path, cropped_objects, classifications)
            if viz_bytes:
                visualization_url = await self.file_handler.save_crop_async(
                    viz_bytes,
                    crop_storage_path,
                    "detection_visualization.jpg"
                )
                print(f"✓ Visualization saved")

        return {'detected_assets': detected_assets, 'visualization_url': visualization_url}

    def process_image(self, image_path, save_crops=False, crop_storage_path=None, iou_threshold=None):
        """Wrapper to run async processing"""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we can't use asyncio.run()
                # This is a limitation - we need to handle this case differently
                print("WARNING: Event loop already running, cannot use asyncio.run()")
                raise RuntimeError("Cannot run async processing when event loop is already running")
            else:
                return asyncio.run(self.process_image_async(image_path, save_crops, crop_storage_path, iou_threshold))
        except RuntimeError as e:
            # No event loop exists, safe to create one
            return asyncio.run(self.process_image_async(image_path, save_crops, crop_storage_path, iou_threshold))
        except Exception as e:
            print(f"Async processing failed: {e}")
            raise e

    def process_uploaded_file(self, uploaded_file, session_uuid=None):
        """Process a Django uploaded file and return detected assets"""
        # Create a temporary file from the uploaded file
        temp_file_path = self.file_handler.create_temp_file(uploaded_file)

        try:
            # Set up storage path for crops
            crop_storage_path = None
            if session_uuid:
                crop_storage_path = f"onboarding/{session_uuid}/detected_assets"

            # Process the image
            result = self.process_image(
                temp_file_path,
                save_crops=True,
                crop_storage_path=crop_storage_path
            )

            detected_assets = result.get('detected_assets', [])
            visualization_url = result.get('visualization_url')

            return {
                'success': True,
                'detected_assets': detected_assets,
                'total_count': len(detected_assets),
                'visualization_url': visualization_url
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'detected_assets': [],
                'total_count': 0,
                'visualization_url': None
            }
        finally:
            # Clean up temporary file
            self.file_handler.cleanup_temp_file(temp_file_path)