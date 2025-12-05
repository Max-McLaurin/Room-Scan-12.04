# Cabinet Deduplication Fix - November 12, 2024

## Problem Statement

**Issue:** Multiple cabinet detections were appearing as separate entries in the asset list when they should be consolidated into a single "Cabinet" entry.

**Example:**
- User screenshot showed **7+ separate cabinet entries** (Cabinet, Cabinet Door, etc.)
- Expected behavior: **1 unified Cabinet entry**

**Root Cause:**
The deduplication logic relied on IoU (Intersection over Union) spatial overlap to determine duplicates. However, individual cabinet doors and sections don't physically overlap in space, resulting in IoU ≈ 0 between detections. This meant the existing singleton logic (IoU threshold of 0.1) couldn't catch non-overlapping cabinet components.

## Original Deduplication Logic

The `AssetDeduplicator` class in `asset_detection/deduplicator.py` used a confidence-based greedy algorithm:

1. Sort detections by confidence (highest first)
2. Keep highest confidence detection
3. Remove duplicates based on:
   - **Semantic matching**: Same normalized classification
   - **IoU threshold**:
     - Singletons: IoU > 0.1
     - Non-singletons: IoU > 0.6

**Limitation:** This approach **requires spatial overlap** to deduplicate. Separate cabinet doors with IoU ≈ 0 would pass through as distinct items.

## Solution: Force Single Entry Mechanism

Implemented a new "force single entry" system that bypasses IoU requirements entirely for specific item types like cabinets.

### Implementation Details

#### 1. Configuration Changes (`asset_detection/config.py`)

Added new `FORCE_SINGLE_ENTRY` list for items that must always consolidate to one entry:

```python
# Items that should ALWAYS be forced to a single entry, regardless of IoU overlap
# This is for distributed items like cabinets where individual sections don't overlap
FORCE_SINGLE_ENTRY = [
    'cabinet', 'cabinets', 'kitchen cabinet', 'cabinet door',
    'cabinetry', 'upper cabinet', 'lower cabinet', 'base cabinet', 'wall cabinet'
]
```

**Key Features:**
- Applied **regardless of spatial overlap**
- Handles all cabinet variations (door, upper, lower, etc.)
- Semantic grouping still maps all variations to 'cabinet'

#### 2. Deduplicator Logic Changes (`asset_detection/deduplicator.py`)

**A. Added `is_force_single_item()` method:**

```python
def is_force_single_item(self, cls):
    """
    Determine if an item should be forced to a single entry regardless of IoU.

    Force-single items are consolidated to one entry even when individual
    detections don't overlap spatially (e.g., multiple cabinet doors).
    """
    normalized = self.normalize_classification(cls)

    # Check if normalized name matches or contains any force-single term
    return normalized in config.FORCE_SINGLE_ENTRY or \
           any(term in normalized for term in config.FORCE_SINGLE_ENTRY)
```

**B. Modified `_should_remove()` method:**

Added force-single check **before** IoU-based deduplication:

```python
# Force single entry items: ALWAYS remove duplicates regardless of IoU
# This handles distributed items like cabinets where sections don't overlap
if self.is_force_single_item(candidate_cls):
    print(f"  Removing force-single duplicate: '{candidate_cls}' (conf={candidate_obj['confidence']:.2f}, IoU={iou:.2f}) - keeping '{current_cls}' (conf={current_obj['confidence']:.2f})")
    return True
```

**Priority Order:**
1. ✅ **Force single entry** (new) - bypasses IoU entirely
2. Singleton check (IoU > 0.1)
3. Non-singleton check (IoU > 0.6)

## Testing

Created comprehensive test (`test_cabinet_dedup.py`) with 7 non-overlapping cabinet detections:

```python
# 7 cabinet detections with different bboxes (zero overlap)
cropped_objects = [
    {'confidence': 0.88, 'bbox': (100, 100, 200, 300)},  # Cabinet
    {'confidence': 0.82, 'bbox': (250, 100, 350, 300)},  # Cabinet
    {'confidence': 0.81, 'bbox': (400, 100, 500, 300)},  # Cabinet
    {'confidence': 0.75, 'bbox': (550, 100, 650, 300)},  # Cabinet Door
    {'confidence': 0.74, 'bbox': (700, 100, 800, 300)},  # Cabinet
    {'confidence': 0.53, 'bbox': (100, 350, 200, 550)},  # Cabinet
    {'confidence': 0.36, 'bbox': (250, 350, 350, 550)},  # Cabinet
]
```

### Test Results

```
✅ SUCCESS: 7 cabinet detections → 1 cabinet entry

Kept cabinet:
  - Cabinet: $300 (conf=0.88)

Deduplication log:
  Removing force-single duplicate: 'Cabinet' (conf=0.82, IoU=0.00)
  Removing force-single duplicate: 'Cabinet' (conf=0.81, IoU=0.00)
  Removing force-single duplicate: 'Cabinet Door' (conf=0.75, IoU=0.00)
  Removing force-single duplicate: 'Cabinet' (conf=0.74, IoU=0.00)
  Removing force-single duplicate: 'Cabinet' (conf=0.53, IoU=0.00)
  Removing force-single duplicate: 'Cabinet' (conf=0.36, IoU=0.00)
```

## Results

✅ **Problem Solved:**
- Multiple cabinet detections now consolidate to **1 unified entry**
- System keeps **highest confidence detection** (0.88 in test)
- Works **regardless of spatial overlap** (IoU = 0.00)

✅ **Behavioral Changes:**
- All cabinet variations (Cabinet, Cabinet Door, Kitchen Cabinet, etc.) → single "Cabinet" entry
- Confidence-based selection: highest confidence cabinet is kept
- Semantic grouping still active: maps all variations to 'cabinet'

## Files Modified

1. **`asset_detection/config.py`** (lines 59-64)
   - Added `FORCE_SINGLE_ENTRY` list
   - Included all cabinet variations

2. **`asset_detection/deduplicator.py`** (lines 110-114, 161-178)
   - Added `is_force_single_item()` method
   - Modified `_should_remove()` to check force-single before IoU

3. **`test_cabinet_dedup.py`** (new file)
   - Comprehensive test for cabinet deduplication
   - Simulates 7 non-overlapping cabinet detections

## Future Enhancements

**Potential Extensions:**
- Add more items to `FORCE_SINGLE_ENTRY` (e.g., countertops, flooring, ceiling)
- Implement value aggregation: sum individual cabinet values for unified entry
- Add configuration option for per-room force-single items
- Support user-defined force-single lists via API

## Usage

The fix is **automatically active** - no configuration changes needed:

1. Start server: `bash start_app.sh`
2. Upload image with multiple cabinet sections
3. System will automatically consolidate to 1 cabinet entry

**Manual Testing:**
```bash
python test_cabinet_dedup.py
```

## Technical Notes

**Performance Impact:** Negligible
- Single additional method call per detection
- O(1) lookup in force-single list
- No additional API calls or heavy computation

**Backward Compatibility:** ✅ Fully compatible
- Existing singleton/non-singleton logic unchanged
- Only adds new force-single layer on top
- No breaking changes to API or data structures

**Edge Cases Handled:**
- Mixed cabinet types (Cabinet + Cabinet Door) → unified
- Different confidence levels → keeps highest
- Various semantic names → all mapped to 'cabinet'
- Zero spatial overlap → force-single handles it

## Summary

Implemented a "force single entry" mechanism that ensures items like cabinets are always consolidated into one entry, regardless of spatial overlap. This solves the issue where multiple non-overlapping cabinet doors/sections appeared as separate entries in the asset list.

**Key Innovation:** Bypassing IoU requirement for specific item types that are inherently distributed across space but should be treated as a single asset.
