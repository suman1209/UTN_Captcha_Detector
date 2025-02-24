import random
import torch
import math
from torchvision.transforms import v2
import torchvision.transforms.functional as F


class Augmentations:
    def __init__(self, config):
        """
        Augmentation class for Captcha dataset, including bounding boxes.

        """
        self.flip_prob = config.flip_prob
        self.flip = v2.RandomHorizontalFlip(p=1.0)
        self.scale_range = config.scale_range
        self.zoom_prob = config.zoom_prob
        self.zoom_range = (1.5, 2.0)
        self.saturation_prob = config.saturation_prob
        self.rotation_prob = config.rotation_prob
        self.rotation_range = (-30, 30)

    def apply(self, image, bboxes, labels):
        """
        Applies augmentations to an image and bounding boxes.

        """
        # Apply horizontal flip
        if random.random() < self.flip_prob:
            image, bboxes = self.horizontal_flip(image, bboxes)

        # Apply zoom
        if random.random() < self.zoom_prob:
            image, bboxes, labels = self.zoom(image, bboxes, labels)

        # Apply saturation changes
        if random.random() < self.saturation_prob:
            image = self.saturation_change(image)

        # Apply rotation
        if random.random() < self.rotation_prob:
            image, bboxes, labels = self.rotate(image, bboxes, labels)

        return image, bboxes, labels

    def horizontal_flip(self, image, bboxes):
        """
        Flips the image and corresponding bounding box horizontally

        Source: (bounding box)
        https://blog.paperspace.com/data-augmentation-for-bounding-boxes/

        """

        img_center = torch.tensor([1 / 2, 0, 1 / 2, 0], dtype=torch.float32)

        image = self.flip(image)

        # Center mirroring transformation
        bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

        box_width = torch.abs(bboxes[:, 0] - bboxes[:, 2])

        # Adjust width
        bboxes[:, 0] -= box_width
        bboxes[:, 2] += box_width

        return image, bboxes

    def zoom(self, image, bboxes, labels, zoom_range=(1.5, 1.8), bbox_in_new_image=0.65):
        """
        Zooms into image by randomly cropping the image and resize to original size.
        Adjust bounding boxes (if < 65% of the bounding box are in the image --> remove)

        """
        zoom_factor = random.uniform(*zoom_range)

        # print(zoom_factor)

        _, height, width = image.shape
        # Crop size in absolute pixels
        crop_height = int(height / zoom_factor)
        crop_width = int(width / zoom_factor)

        # Random cropping center
        center_x = random.randint(crop_width // 2, width - crop_width // 2)
        center_y = random.randint(crop_height // 2, height - crop_height // 2)

        # Crop box (not exceeding boundaries)
        crop_x_min = max(0, center_x - crop_width // 2)
        crop_y_min = max(0, center_y - crop_height // 2)

        image = F.crop(image, crop_y_min, crop_x_min, crop_height, crop_width)

        new_bboxes = []
        new_labels = []

        # Adjust bboxes and labels:
        for bbox, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = bbox

            # Convert to absolute values
            abs_x_min = x_min * width
            abs_y_min = y_min * height
            abs_x_max = x_max * width
            abs_y_max = y_max * height

            # Original box area
            orig_bbox_area = (abs_x_max - abs_x_min) * (abs_y_max - abs_y_min)

            # Adjust for crop
            intersection_x_min = max(abs_x_min, crop_x_min)
            intersection_y_min = max(abs_y_min, crop_y_min)
            intersection_x_max = min(abs_x_max, crop_x_min + crop_width)
            intersection_y_max = min(abs_y_max, crop_y_min + crop_height)

            # Intersection area
            intersection_width = max(0, intersection_x_max - intersection_x_min)
            intersection_height = max(0, intersection_y_max - intersection_y_min)
            intersection_area = intersection_width * intersection_height

            # Check if at least 65% of original box is in cropped area
            # --> only detect boxes where object is visible in the new area
            if intersection_area / orig_bbox_area >= bbox_in_new_image:
                # Clip to stay within bounds
                intersection_x_min = max(0, min(intersection_x_min - crop_x_min, crop_width))
                intersection_y_min = max(0, min(intersection_y_min - crop_y_min, crop_height))
                intersection_x_max = max(0, min(intersection_x_max - crop_x_min, crop_width))
                intersection_y_max = max(0, min(intersection_y_max - crop_y_min, crop_height))

                # Normalize to new crop dim
                new_bboxes.append([
                    intersection_x_min / crop_width,
                    intersection_y_min / crop_height,
                    intersection_x_max / crop_width,
                    intersection_y_max / crop_height
                ])
                new_labels.append(label)

        new_bboxes = torch.tensor(new_bboxes, dtype=torch.float32)
        new_labels = torch.tensor(new_labels, dtype=torch.long)

        # Resize to original dimensions
        image = F.resize(image, (height, width))

        return image, new_bboxes, new_labels

    def saturation_change(self, image, saturation_lowest=1.8, saturation_highest=3):
        """
        Change the saturation of the image.
        BBoxes stay the same.
        """

        # TODO:
        # get images in RGB to change saturation properly
        # de-normalize?

        saturation_fac = random.uniform(saturation_lowest, saturation_highest)

        image = F.adjust_saturation(image, saturation_fac)

        return image

    def rotate(self, image, bboxes, labels, bbox_in_new_image=1.0):
        """
        Rotate the image and the bounding boxes.
        Adjust bounding boxes (if < 100% of the bounding box are in the image --> remove)

        idea: https://www.digitalocean.com/community/tutorials/data-augmentation-for-object-detection-rotation-and-shearing
        """

        angle = random.uniform(*self.rotation_range)

        _, height, width = image.shape

        image = F.rotate(image, angle)

        image_center = torch.tensor([width / 2, height / 2], dtype=torch.float32)

        # Rotation matrix
        angle_radian = math.radians(angle)
        cos_a, sin_a = math.cos(angle_radian), math.sin(angle_radian)

        new_bboxes = []
        new_labels = []

        for bbox, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = bbox.tolist()

            # Convert to absolute values
            abs_x_min = x_min * width
            abs_y_min = y_min * height
            abs_x_max = x_max * width
            abs_y_max = y_max * height

            # Original box area
            orig_bbox_area = (abs_x_max - abs_x_min) * (abs_y_max - abs_y_min)

            # Corner points
            corners = torch.tensor([
                [abs_x_min, abs_y_min],
                [abs_x_max, abs_y_min],
                [abs_x_max, abs_y_max],
                [abs_x_min, abs_y_max]
            ], dtype=torch.float32)

            # Shift to image center
            corners -= image_center

            # Apply rotation
            rotated_corners = torch.mm(corners, torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32))

            # Shift back to original position
            rotated_corners += image_center

            # New bounding box
            new_x_min = rotated_corners[:, 0].min().item()
            new_y_min = rotated_corners[:, 1].min().item()
            new_x_max = rotated_corners[:, 0].max().item()
            new_y_max = rotated_corners[:, 1].max().item()

            # Clip to stay within image
            intersection_x_min = max(0, new_x_min)
            intersection_y_min = max(0, new_y_min)
            intersection_x_max = min(width, new_x_max)
            intersection_y_max = min(height, new_y_max)

            # Intersection area
            inter_width = max(0, intersection_x_max - intersection_x_min)
            inter_height = max(0, intersection_y_max - intersection_y_min)
            intersection_area = inter_width * inter_height

            # Check if 100% of the original bbox is in rotated area
            if intersection_area / orig_bbox_area >= bbox_in_new_image:
                # Normalize bbox
                new_bboxes.append([
                    intersection_x_min / width,
                    intersection_y_min / height,
                    intersection_x_max / width,
                    intersection_y_max / height
                ])
                new_labels.append(label)

        new_bboxes = torch.tensor(new_bboxes, dtype=torch.float32)
        new_labels = torch.tensor(new_labels, dtype=torch.long)

        return image, new_bboxes, new_labels
