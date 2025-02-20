# source https://gist.github.com/tuxedocat/fb024dfa36648797084d

import numpy as np

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


    import torch
import numpy as np
from src_code.data_utils.dataset_utils import get_dataloader


def detect_objects(model, image):
    # Placeholder function to detect objects. 
    # returns  a list of (character, bounding_box) tuples.

    pred_locs, pred_classes = model(image.unsqueeze(0))  # Forward pass
    # Apply non-maximum suppression (NMS) to get final boxes
    return [(char, bbox) for char, bbox in zip(pred_classes, pred_locs)]



def edit_score(model, dataloader):
    # Compute edit distance from model predictions and ground truths
    # takes in the trained model and images and GT Are passed through data loader
    # returns a list of edit distances per image.


    edit_distances = []
    
    for image, groundtruth_text in dataloader:
        batch_edit_distances = []

        for i in range(len(images)):
        
            # should date on 1 single image
            detected_objects = detect_objects(model, image)  # Detect characters & bounding boxes
            
            if not detected_objects:
                predicted_text = ""  # for when No characters are detected
            else:
                # Sorting detected characters by BB x coordinates
                detected_objects.sort(key=lambda obj: obj[1][0])  # Sort by x_min of bbox
                predicted_text = "".join(char for char, _ in detected_objects)
            
            # Compute edit distance
            edit_dist = levenshtein(predicted_text, groundtruth_text)
            edit_distances.append(edit_dist)

    return edit_distances

