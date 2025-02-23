import random
import torch
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
            x1, y1, x2, y2 = bbox

            # Convert to absolute values
            abs_x1 = x1 * width
            abs_y1 = y1 * height
            abs_x2 = x2 * width
            abs_y2 = y2 * height

            # Original box area
            orig_bbox_area = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)

            # Adjust for crop
            intersection_x1 = max(abs_x1, crop_x_min)
            intersection_y1 = max(abs_y1, crop_y_min)
            intersection_x2 = min(abs_x2, crop_x_min + crop_width)
            intersection_y2 = min(abs_y2, crop_y_min + crop_height)

            # Intersection area
            intersection_width = max(0, intersection_x2 - intersection_x1)
            intersection_height = max(0, intersection_y2 - intersection_y1)
            intersection_area = intersection_width * intersection_height

            # Check if at least 65% of original box is in cropped area
            # --> only detect boxes where object is visible in the new area
            if intersection_area / orig_bbox_area >= bbox_in_new_image:
                # Clip to stay within bounds
                intersection_x1 = max(0, min(intersection_x1 - crop_x_min, crop_width))
                intersection_y1 = max(0, min(intersection_y1 - crop_y_min, crop_height))
                intersection_x2 = max(0, min(intersection_x2 - crop_x_min, crop_width))
                intersection_y2 = max(0, min(intersection_y2 - crop_y_min, crop_height))

                # Normalize to new crop dim
                new_bboxes.append([
                    intersection_x1 / crop_width,
                    intersection_y1 / crop_height,
                    intersection_x2 / crop_width,
                    intersection_y2 / crop_height
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
