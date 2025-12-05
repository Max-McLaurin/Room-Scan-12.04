"""
Test cabinet deduplication logic
"""
import sys
sys.path.insert(0, '/Users/maxmclaurin/Documents/PARA/Projects/AI_Detect2')

from asset_detection.deduplicator import AssetDeduplicator

# Create test data simulating multiple cabinet detections with different bboxes (no overlap)
cropped_objects = [
    {'confidence': 0.88, 'bbox': (100, 100, 200, 300), 'detection_id': 'c1'},
    {'confidence': 0.82, 'bbox': (250, 100, 350, 300), 'detection_id': 'c2'},
    {'confidence': 0.81, 'bbox': (400, 100, 500, 300), 'detection_id': 'c3'},
    {'confidence': 0.75, 'bbox': (550, 100, 650, 300), 'detection_id': 'c4'},
    {'confidence': 0.74, 'bbox': (700, 100, 800, 300), 'detection_id': 'c5'},
    {'confidence': 0.53, 'bbox': (100, 350, 200, 550), 'detection_id': 'c6'},
    {'confidence': 0.36, 'bbox': (250, 350, 350, 550), 'detection_id': 'c7'},
]

classifications = [
    {'name': 'Cabinet', 'category': 'Kitchen', 'estimated_value': '$300'},
    {'name': 'Cabinet', 'category': 'Kitchen', 'estimated_value': '$300'},
    {'name': 'Cabinet', 'category': 'Kitchen', 'estimated_value': '$300'},
    {'name': 'Cabinet Door', 'category': 'Kitchen', 'estimated_value': '$150'},
    {'name': 'Cabinet', 'category': 'Kitchen', 'estimated_value': '$500'},
    {'name': 'Cabinet', 'category': 'Kitchen', 'estimated_value': '$300'},
    {'name': 'Cabinet', 'category': 'Kitchen', 'estimated_value': '$300'},
]

print("=" * 80)
print("Testing Cabinet Deduplication Logic")
print("=" * 80)
print(f"\nInput: {len(cropped_objects)} cabinet detections with NO spatial overlap")
print("\nDetections:")
for i, (obj, cls) in enumerate(zip(cropped_objects, classifications), 1):
    print(f"  {i}. {cls['name']} (confidence={obj['confidence']:.2f}, bbox={obj['bbox']})")

# Run deduplication
dedup = AssetDeduplicator()
kept_objects, kept_classifications = dedup.remove_duplicates(cropped_objects, classifications)

print(f"\n{'='*80}")
print(f"Result: {len(kept_objects)} cabinet entries after deduplication")
print(f"{'='*80}")

if len(kept_objects) == 1:
    print("✅ SUCCESS: Only 1 cabinet entry (as expected)")
    print(f"\nKept cabinet:")
    print(f"  - {kept_classifications[0]['name']}: {kept_classifications[0]['estimated_value']} (conf={kept_objects[0]['confidence']:.2f})")
else:
    print(f"❌ FAILED: Found {len(kept_objects)} cabinet entries, expected 1")
    print(f"\nAll kept items:")
    for i, (obj, cls) in enumerate(zip(kept_objects, kept_classifications), 1):
        print(f"  {i}. {cls['name']}: {cls['estimated_value']} (conf={obj['confidence']:.2f})")

print(f"\n{'='*80}")
