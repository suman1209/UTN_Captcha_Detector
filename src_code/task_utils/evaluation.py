# source https://gist.github.com/tuxedocat/fb024dfa36648797084d

import numpy as np
from src_code.data_utils.dataset_utils import get_dataloader
from src_code.model_utils.utils import find_IoU


def levenshtein(x, y):

    # Levenshtein distance (edit distance) between two strings
    x = np.array(list(x))
    y = np.array(list(y))

    # Initialize DP table
    D = np.zeros((len(x) + 1, len(y) + 1), dtype=int)
    D[0, 1:] = np.arange(1, len(y) + 1)
    D[1:, 0] = np.arange(1, len(x) + 1)

    # Fill the table
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            D[i, j] = min(
                D[i - 1, j - 1] + cost,  # Substitution
                D[i, j - 1] + 1,         # Insertion
                D[i - 1, j] + 1          # Deletion
            )
    return D[-1, -1]


def detect_objects(model, image):
    # Placeholder function to detect objects.
    # returns  a list of (character, bounding_box) tuples.
    device = next(model.parameters()).device  # Get model's device
    image = image.to(device)
    pred_locs, pred_classes = model(image.unsqueeze(0))  # Forward pass
    # Apply NMS to get final boxes
    return [(char, bbox) for char, bbox in zip(pred_classes, pred_locs)]


def edit_score(model, dataloader):
    # Compute edit distance from model predictions and ground truths
    # takes in the trained model and images and GT Are passed
    # through data loader
    # returns a list of edit distances per image.

    edit_distances = []

    for images, groundtruth_text in dataloader:

        batch_edit_distances = []

        for i in range(len(images)):

            # should date on 1 single image
            # Detect characters & bounding boxes
            detected_objects = detect_objects(model, images[i])

            if not detected_objects:
                predicted_text = ""  # for when No characters are detected
            else:
                # Sorting detected characters by BB x coordinates
                # Sort by x_min of bbox
                detected_objects.sort(key=lambda obj: obj[1][0])
                predicted_text = "".join(char for char, _ in detected_objects)

            # Compute edit distance
            edit_dist = levenshtein(predicted_text, groundtruth_text)
            batch_edit_distances.append(edit_dist)

        edit_distances.extend(batch_edit_distances)

    return edit_distances


# next compute precision & recall for a single class
# it takes as arguments detections (list of predicted boxes)
#   list of ground truth boxes and IoU threshold.
# returns precision list and recall list to be used to compute AP

def compute_precision_recall(detections, ground_truths, iou_threshold=0.5):
    # Sort by confidence score
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    true_positives = np.zeros(len(detections))
    false_positives = np.zeros(len(detections))
    total_gt = len(ground_truths)
    matched_gts = set()

    for i, (confidence, pred_box) in enumerate(detections):
        best_iou = 0
        best_gt_idx = -1

        for j, gt_box in enumerate(ground_truths):
            iou = find_IoU(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx not in matched_gts:
            true_positives[i] = 1
            matched_gts.add(best_gt_idx)
        else:
            false_positives[i] = 1

    cumulative_true_positives = np.cumsum(true_positives)
    cumulative_false_positives = np.cumsum(false_positives)
    precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives + 1e-8)
    recall = cumulative_true_positives / total_gt if total_gt else np.zeros(len(detections))

    return precision, recall


def compute_average_precision(precision, recall):

    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    average_precision = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return average_precision


# compute mAP
def compute_map(predictions, ground_truths, num_classes):

    aps = []

    for class_id in range(num_classes):
        if class_id not in predictions:
            continue

        precision, recall = compute_precision_recall(predictions[class_id], ground_truths.get(class_id, []))
        average_precision = compute_average_precision(precision, recall)
        aps.append(average_precision)

    return np.mean(aps) if aps else 0


# source https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py

# takes boxes, confidence scores and iou threshold computed above
# to return a list of intices of the kept boxes

def non_max_supression(boxes, scores, iou_threshold=0.5):

    if len(boxes) == 0:
        return []

    # Convert to NumPy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Sort by confidence score
    sorted_indices = np.argsort(scores)[::-1]
    kept_boxes = []

    while len(sorted_indices) > 0:
        # Pick the onesthat have the highest confidence
        current_idx = sorted_indices[0]
        kept_boxes.append(current_idx)

        if len(sorted_indices) == 1:
            break

        # Compute IoU with remaining boxes
        ious = np.array([find_IoU(boxes[current_idx], boxes[i]) for i in sorted_indices[1:]])

        # Keep only boxes those that IoU < threshold
        remaining_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[1:][remaining_indices]

    return kept_boxes


# apply nms per class to detect multiple classes (letter and numbers)
# takes predictions and iou threshold
# to return a dictionary for filtered boxes and scores

def nms_per_class(predictions, iou_threshold=0.5):

    filtered_predictions = {}

    for class_id, (boxes, scores) in predictions.items():
        if len(boxes) == 0:
            continue

        keep_indices = non_max_supression(boxes, scores, iou_threshold)
        filtered_predictions[class_id] = ([boxes[i] for i in keep_indices], [scores[i] for i in keep_indices])

    return filtered_predictions
