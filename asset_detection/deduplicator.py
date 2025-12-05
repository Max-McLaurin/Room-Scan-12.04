"""
Asset deduplication logic

This module handles the removal of duplicate and semantically similar asset detections
based on IoU overlap and semantic classification matching.
"""

from .bbox_utils import calculate_iou
from . import config


class AssetDeduplicator:
    """
    Handles deduplication of detected assets based on:
    1. Semantic similarity (e.g., "Faucet" matches "Faucet")
    2. IoU overlap (configurable thresholds for singletons vs non-singletons)
    3. Confidence-based prioritization (keeps higher confidence detections)
    """

    def remove_duplicates(self, cropped_objects, classifications):
        """
        Remove semantic duplicates from detected assets.

        Args:
            cropped_objects: List of detected object data (bbox, confidence, etc.)
            classifications: List of OpenAI classifications for each object

        Returns:
            Tuple of (deduplicated_objects, deduplicated_classifications)
        """
        if len(cropped_objects) == 0:
            return cropped_objects, classifications

        # Combine data for processing
        combined = list(zip(cropped_objects, classifications))

        # Filter out very low confidence detections first
        combined = self._filter_low_confidence(combined)

        # Sort by confidence (highest first)
        combined = sorted(combined, key=lambda x: x[0]['confidence'], reverse=True)

        # Perform deduplication
        kept_items = self._deduplicate_items(combined)

        # Unzip the results
        if kept_items:
            kept_objects, kept_classifications = zip(*kept_items)
            return list(kept_objects), list(kept_classifications)
        else:
            return [], []

    def _filter_low_confidence(self, combined):
        """Filter out detections below minimum confidence threshold."""
        return [(obj, cls) for obj, cls in combined if obj['confidence'] >= config.MIN_CONFIDENCE_THRESHOLD]

    def _deduplicate_items(self, combined):
        """
        Main deduplication loop - removes duplicates based on semantic similarity and IoU.

        Uses a greedy algorithm:
        1. Keep the highest confidence detection
        2. Remove all similar/overlapping detections based on item type
        3. Repeat with next highest confidence detection
        """
        keep = []

        while combined:
            # Keep the highest confidence detection
            current_obj, current_cls = combined.pop(0)
            keep.append((current_obj, current_cls))

            current_normalized = self.normalize_classification(current_cls)
            current_is_singleton = self.is_singleton_item(current_cls)

            # Remove semantically similar and overlapping detections
            filtered = []
            for obj, cls in combined:
                if not self._should_remove(current_obj, current_cls, current_normalized,
                                          current_is_singleton, obj, cls):
                    filtered.append((obj, cls))

            combined = filtered

        return keep

    def _should_remove(self, current_obj, current_cls, current_normalized,
                       current_is_singleton, candidate_obj, candidate_cls):
        """
        Determine if a candidate detection should be removed as a duplicate.

        Args:
            current_obj: The reference object (highest confidence so far)
            current_cls: Classification of reference object
            current_normalized: Normalized classification of reference object
            current_is_singleton: Whether reference object is a singleton type
            candidate_obj: Candidate object to check for removal
            candidate_cls: Classification of candidate object

        Returns:
            True if candidate should be removed, False otherwise
        """
        normalized = self.normalize_classification(candidate_cls)
        iou = calculate_iou(current_obj['bbox'], candidate_obj['bbox'])

        # Only check items with same semantic classification
        if normalized != current_normalized:
            return False

        # Force single entry items: ALWAYS remove duplicates regardless of IoU
        # This handles distributed items like cabinets where sections don't overlap
        if self.is_force_single_item(candidate_cls):
            print(f"  Removing force-single duplicate: '{candidate_cls}' (conf={candidate_obj['confidence']:.2f}, IoU={iou:.2f}) - keeping '{current_cls}' (conf={current_obj['confidence']:.2f})")
            return True

        # Different deduplication strategies based on item type
        if current_is_singleton:
            # Singleton items: remove any duplicate with same semantic class
            # Use lower IoU threshold since these should be unique
            if iou > config.SINGLETON_IOU_THRESHOLD:
                print(f"  Removing singleton duplicate: '{candidate_cls}' (conf={candidate_obj['confidence']:.2f}, IoU={iou:.2f}) - keeping '{current_cls}' (conf={current_obj['confidence']:.2f})")
                return True
        else:
            # Non-singleton items: only remove if high overlap
            # This allows multiple items to coexist
            if iou > config.NON_SINGLETON_IOU_THRESHOLD:
                print(f"  Removing overlapping duplicate: '{candidate_cls}' (conf={candidate_obj['confidence']:.2f}, IoU={iou:.2f}) - keeping '{current_cls}' (conf={current_obj['confidence']:.2f})")
                return True

        return False

    def normalize_classification(self, cls):
        """
        Normalize classification for comparison.

        Handles both dict and string formats, and maps semantically similar items
        to a common name using the semantic groups from config.

        Args:
            cls: Classification (dict with 'name' key or string)

        Returns:
            Normalized classification string (lowercase)
        """
        # Handle both dict and string formats
        if isinstance(cls, dict):
            name = cls.get('name', 'Unknown Item')
        else:
            name = str(cls)

        # Remove plurals and common variations
        normalized = name.lower().strip()

        # Map semantically similar items using config
        for group_name, variants in config.SEMANTIC_GROUPS.items():
            if any(variant in normalized for variant in variants):
                return group_name

        return normalized

    def is_force_single_item(self, cls):
        """
        Determine if an item should be forced to a single entry regardless of IoU.

        Force-single items are consolidated to one entry even when individual
        detections don't overlap spatially (e.g., multiple cabinet doors).

        Args:
            cls: Classification (dict with 'name' key or string)

        Returns:
            True if item should be forced to single entry, False otherwise
        """
        normalized = self.normalize_classification(cls)

        # Check if normalized name matches or contains any force-single term
        return normalized in config.FORCE_SINGLE_ENTRY or \
               any(term in normalized for term in config.FORCE_SINGLE_ENTRY)

    def is_singleton_item(self, cls):
        """
        Determine if an item should typically be unique in a room.

        Singleton items will be aggressively deduplicated using a low IoU threshold.

        Args:
            cls: Classification (dict with 'name' key or string)

        Returns:
            True if item is a singleton type, False otherwise
        """
        normalized = self.normalize_classification(cls)
        return normalized in config.SINGLETON_ITEM_TYPES or any(s in normalized for s in config.SINGLETON_ITEM_TYPES)
