"""
Classification orchestrator

This module handles orchestrating OpenAI classification and categorization
for detected assets using async/concurrent API calls.
"""

import asyncio
import aiohttp


class ClassificationOrchestrator:
    """
    Orchestrates async OpenAI API calls for asset classification and categorization.

    Responsibilities:
    - Manage concurrent classification requests
    - Extract asset names from classifications
    - Coordinate batch categorization
    - Handle aiohttp session lifecycle
    """

    def __init__(self, image_processor):
        """
        Initialize the orchestrator with required dependencies.

        Args:
            image_processor: ImageProcessor instance for image conversion
        """
        from .classifier import AssetClassifier

        self.classifier = AssetClassifier()
        self.image_processor = image_processor

    async def classify_and_categorize(self, cropped_objects):
        """
        Classify all detected objects concurrently using OpenAI.

        This is the main entry point that:
        1. Classifies all objects using OpenAI (concurrent)
        2. Returns classifications (categorization removed for standalone mode)

        Args:
            cropped_objects: List of detected object data with images

        Returns:
            Tuple of (classifications, categorizations) - categorizations will be None
        """
        print(f"Found {len(cropped_objects)} objects, classifying with OpenAI...")

        # Create aiohttp session for concurrent requests
        async with aiohttp.ClientSession() as session:
            # Classify all objects concurrently
            classifications = await self._classify_all(cropped_objects, session)

            # No categorization in standalone mode
            categorizations = None

        return classifications, categorizations

    async def _classify_all(self, cropped_objects, session):
        """
        Classify all detected objects concurrently using OpenAI.

        Args:
            cropped_objects: List of detected object data with images
            session: aiohttp ClientSession for API calls

        Returns:
            List of OpenAI classification results
        """
        # Create tasks for all OpenAI classifications
        tasks = []
        for i, obj in enumerate(cropped_objects):
            base64_image = self.image_processor.to_base64(obj['image'])
            task = self.classifier.classify_async(session, base64_image, i+1)
            tasks.append(task)

        # Execute all OpenAI calls concurrently
        classifications = await asyncio.gather(*tasks)

        return classifications

    async def _categorize_all(self, classifications, session):
        """
        Categorize all classified assets in a single batch request.

        Args:
            classifications: List of OpenAI classification results
            session: aiohttp ClientSession for API calls

        Returns:
            List of categorization results
        """
        print("Categorizing detected assets in batch...")

        # Extract asset names from classifications
        asset_names = self._extract_asset_names(classifications)

        # Single API call for all categorizations
        categorizations = await self.classifier.categorize_batch_async(session, asset_names)

        return categorizations

    @staticmethod
    def _extract_asset_names(classifications):
        """
        Extract asset names from OpenAI classification results.

        Handles both dict responses (with 'name' key) and string fallback.

        Args:
            classifications: List of OpenAI classification results

        Returns:
            List of asset name strings
        """
        asset_names = [
            classification.get('name', 'Unknown Item') if isinstance(classification, dict) else str(classification)
            for classification in classifications
        ]
        return asset_names
