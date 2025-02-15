import random
import torch
from torchvision.transforms import v2


class Augmentations:
    def __init__(self, flip_prob=0.3, scale_range=(0.8, 1.2)):
        """
        Augmentation class for Captcha dataset, including bounding boxes.

        """
        self.flip_prob = flip_prob
        self.flip = v2.RandomHorizontalFlip(p=1.0)
        self.scale_range = scale_range

    def apply(self, image, bboxes):
        """
        Applies augmentations to an image and bounding boxes.

        """
        # Apply horizontal flip
        if random.random() < self.flip_prob:
            image, bboxes = self.horizontal_flip(image, bboxes)

        # Apply scaling
        # TODO: fix issue with dataloader
        # if random.random() < 0.5:
        #     image, bboxes = self.scale(image, bboxes)

        return image, bboxes

    def horizontal_flip(self, image, bboxes):
        """
        Flips the image and corresponding bounding box horizontally

        Source: (bounding box)
        https://blog.paperspace.com/data-augmentation-for-bounding-boxes/

        """

        image_width = image.shape[-1]

        img_center = torch.tensor([image_width / 2, 0, image_width / 2, 0], dtype=torch.float32)

        image = self.flip(image)

        # Center mirroring transformation
        bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

        box_width = torch.abs(bboxes[:, 0] - bboxes[:, 2])

        # Adjust width
        bboxes[:, 0] -= box_width
        bboxes[:, 2] += box_width

        return image, bboxes

    def scale(self, image, bboxes):
        """
        Scales the image and bounding boxes

        """
        scale_factor = random.uniform(*self.scale_range)
        image_height, image_width = image.shape[-2:]

        # Resize image
        new_width = int(image_width * scale_factor)
        new_height = int(image_height * scale_factor)
        resize_transform = v2.Resize((new_height, new_width))
        image = resize_transform(image)

        # Resize bounding box
        bboxes *= scale_factor

        return image, bboxes
