"""Pure mathematical functions for bounding box geometry calculations"""


def calculate_intersection_area(box1, box2):
    """Calculate the intersection area between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    return (x_right - x_left) * (y_bottom - y_top)


def calculate_box_area(box):
    """Calculate the area of a bounding box"""
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    intersection = calculate_intersection_area(box1, box2)
    
    box1_area = calculate_box_area(box1)
    box2_area = calculate_box_area(box2)
    union_area = box1_area + box2_area - intersection
    
    return intersection / union_area if union_area > 0 else 0.0


def calculate_bbox_center(bbox):
    """Calculate the center point of a bounding box"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y


def calculate_square_size(bbox):
    """Calculate the size for a square crop based on bbox dimensions"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return max(width, height)


def calculate_square_crop_bounds(bbox, image_shape):
    """Calculate square crop boundaries centered on the bounding box"""
    size = calculate_square_size(bbox)
    center_x, center_y = calculate_bbox_center(bbox)
    
    half_size = size // 2
    crop_x1 = max(0, center_x - half_size)
    crop_y1 = max(0, center_y - half_size)
    crop_x2 = min(image_shape[1], center_x + half_size)
    crop_y2 = min(image_shape[0], center_y + half_size)
    
    return crop_y1, crop_y2, crop_x1, crop_x2


def remove_duplicates_by_iou(detections, iou_threshold=0.5):
    """Remove duplicate detections based on IoU threshold (NMS algorithm)"""
    if len(detections) == 0:
        return detections
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    remaining = sorted_detections.copy()
    
    while remaining:
        # Keep the highest confidence detection
        current = remaining.pop(0)
        keep.append(current)
        
        # Remove all detections that overlap significantly with current
        remaining = [
            det for det in remaining
            if calculate_iou(current['bbox'], det['bbox']) < iou_threshold
        ]
    
    return keep

