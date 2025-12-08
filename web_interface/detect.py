from ultralytics import YOLO
import os

def calculate_iou(box1, box2):
    """
    Calculate IoU for two YOLO bounding boxes in normalized format.
    box1, box2: [center_x, center_y, width, height]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corners
    x1_left = x1 - w1 / 2
    y1_top = y1 - h1 / 2
    x1_right = x1 + w1 / 2
    y1_bottom = y1 + h1 / 2

    x2_left = x2 - w2 / 2
    y2_top = y2 - h2 / 2
    x2_right = x2 + w2 / 2
    y2_bottom = y2 + h2 / 2

    # Intersection
    x_left = max(x1_left, x2_left)
    y_top = max(y1_top, y2_top)
    x_right = min(x1_right, x2_right)
    y_bottom = min(y1_bottom, y2_bottom)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def detect_person_without_helmet(results, iou_threshold=0.5):
    warnings = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        # Assume boxes.data is tensor [xyxy, conf, cls]
        detections = []
        for box in boxes.data:
            x_center = (box[0] + box[2]) / 2 / result.orig_shape[1]  # normalize
            y_center = (box[1] + box[3]) / 2 / result.orig_shape[0]
            width = (box[2] - box[0]) / result.orig_shape[1]
            height = (box[3] - box[1]) / result.orig_shape[0]
            conf = box[4]
            cls = int(box[5])
            detections.append({'center_x': x_center, 'center_y': y_center, 'width': width, 'height': height, 'class': cls})

        helmets = [d for d in detections if d['class'] == 0]  # helmet
        heads = [d for d in detections if d['class'] == 1]  # head
        persons = [d for d in detections if d['class'] == 2]  # person

        for person in persons:
            # Check if person has a helmet (IoU with any helmet > threshold)
            has_helmet = any(calculate_iou(
                [person['center_x'], person['center_y'], person['width'], person['height']],
                [h['center_x'], h['center_y'], h['width'], h['height']]
            ) > iou_threshold for h in helmets)
            if not has_helmet:
                warnings.append("WARNING: Person without helmet detected!")

        # Also check heads without helmets
        for head in heads:
            has_helmet = any(calculate_iou(
                [head['center_x'], head['center_y'], head['width'], head['height']],
                [h['center_x'], h['center_y'], h['width'], h['height']]
            ) > iou_threshold for h in helmets)
            if not has_helmet:
                warnings.append("WARNING: Head without helmet detected!")

    return warnings

def check_image(image_path, model_path='../runs/detect/train/weights/best.pt'):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return []

    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.25, save=False, verbose=False)

    warnings = detect_person_without_helmet(results)
    return warnings
