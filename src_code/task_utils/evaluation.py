# source https://gist.github.com/tuxedocat/fb024dfa36648797084d
import os
import numpy as np
import json
from src_code.model_utils.utils import find_IoU
import torch
from tqdm import tqdm

def generate_edit_distance(model, val_loader, configs):
        """
        Generates captchas for the test set using the pre-loaded model and saves the results in a JSON file.
        """
        edit_distances = []
        with torch.no_grad():
            for images, gt_truth, labels_gt in tqdm(val_loader):
                images = images.to(configs.device)
                for idx, image in enumerate(images):
                    # print(image.unsqueeze(0))
                    loc_preds, cls_preds, _ = model(image.unsqueeze(0))
                    boxes, labels, scores = model.detect_object(loc_preds, cls_preds, min_score=configs.nms_min_cls_score,
                     max_overlap=configs.nms_iou_score, top_k=configs.nms_topk)
                    
                    list_boxes = boxes[0].tolist()
                    assert len(list_boxes) == len(labels[0])
                    for i, label_idx in enumerate(labels[0].tolist()):
                        list_boxes[i].append(label_idx)
                    list_boxes = sorted(list_boxes, key=lambda x: x[0])
                    predicted_captcha = "".join([configs.category_id_labels[i[-1]] for i in list_boxes])
                    gt_string = "".join([configs.category_id_labels[i] for i in labels_gt[idx].tolist()])
                    edit_distance = levenshtein(gt_string, predicted_captcha)
                    edit_distances.append(edit_distance)
        mean_edit_distance, captcha_count = np.mean(np.array(edit_distances)), len(edit_distances)
        return mean_edit_distance.item(), captcha_count


def test_generate_captchas_submission(model, test_loader, configs, test_path, output_file="test_captchas_submission.json"):
    """
    Generates captchas for the test set using the pre-loaded model and saves the results in a JSON file.
    Image IDs are now taken directly from the filename (without the file extension).
    """
    results = []

    # Ensure filenames are sorted numerically
    filenames = sorted(
        [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))],
        key=lambda x: int(os.path.splitext(x)[0])  # Sort based on numeric value
    )

    with torch.no_grad():
        for batch_idx, images in tqdm(enumerate(test_loader)):
            images = images.to(configs.device)

            for idx, image in enumerate(images):
                # Stop if there are no more filenames left
                if idx + batch_idx * len(images) >= len(filenames):
                    print(f"All images processed up to index {idx + batch_idx * len(images)}. Stopping.")
                    break

                filename = filenames[idx + batch_idx * len(images)]  # Get file based on current index
                file_number = os.path.splitext(filename)[0]  # Extract number without file extension
                image_id = file_number  # Use file number as image_id

                # print(f"Processing: {file_number} → Image ID: {image_id}")  # Debugging output

                # Ensure image has correct dimensions
                if image.dim() == 2:
                    image = image.unsqueeze(0)  # Convert to (1, H, W)
                if image.dim() == 3:
                    image = image.unsqueeze(0)  # Convert to (1, C, H, W)

                # Forward pass through model
                loc_preds, cls_preds, _ = model(image)
                boxes, labels, scores = model.detect_object(
                    loc_preds, cls_preds, min_score=0.15, max_overlap=0.5, top_k=20
                )

                list_boxes = boxes[0].tolist()
                
                if len(list_boxes) == 0:
                    print(f"⚠ Warning: No predictions for {image_id}, skipping.")
                    continue  # Skip images with no predictions

                assert len(list_boxes) == len(labels[0])

                # Append label to box data
                for i, label_idx in enumerate(labels[0].tolist()):
                    list_boxes[i].append(label_idx)

                # Sort boxes left-to-right
                list_boxes = sorted(list_boxes, key=lambda x: x[0])

                # Convert labels to string
                predicted_captcha = "".join([configs.category_id_labels[i[-1]] for i in list_boxes])

                results.append({
                    "image_id": image_id,  # Now using the file number directly
                    "captcha_string": predicted_captcha
                })

    # Debugging output to verify all images are included
    # print(f"📌 Processed images: {[r['image_id'] for r in results]}")
    assert len(results) == len(filenames), "⚠ Not all images were processed!"

    # Save results to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Predictions saved to {output_file}")

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
    return D[-1, -1].item()


def detect_objects(model, image):
    # returns  a list of (character, bounding_box) tuples.
    device = next(model.parameters()).device  # Get model's device
    image = image.to(device)
    model = model.to(device)
    pred_locs, pred_classes, _ = model(image.unsqueeze(0))  # Forward pass
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
