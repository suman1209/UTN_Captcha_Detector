import torch
import os
from torchvision.datasets import VisionDataset
from .augmentation import Augmentations
from .preprocessing import get_img_transform
from PIL import Image

import copy

category_id_labels = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z"
}

# reverse lookup
label_map = {v: k for k, v in category_id_labels.items()}
label_map['background'] = 36


class CaptchaDataset(VisionDataset):
    def __init__(self, preprocessed_dir, labels_dir=None, augment=True, config=None, img_transform=None):
        """
        Captcha Dataset for loading preprocessed data

        preprocessed_dir: preprocessed tensors
        labels_dir: label files
        """
        super().__init__(preprocessed_dir)

        self.preprocessed_dir = preprocessed_dir
        self.labels_dir = labels_dir
        self.augment = augment
        self.augmentations = Augmentations(config) if augment else None

        self.image_names = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith((".pt", ".png"))])
        self.img_transform = img_transform
        
    def load_labels(self, image_name):
        """
        Loading bounding boxes and class labels
        """

        label_file = os.path.join(self.labels_dir, image_name.replace(".png", ".txt"))

        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        bboxes = []
        labels = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                class_label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                x_min = max(0, x_center - width / 2)
                y_min = max(0, y_center - height / 2)
                x_max = min(1, x_center + width / 2)
                y_max = min(1, y_center + height / 2)

                bboxes.append([x_min, y_min, x_max, y_max])
                labels.append(class_label)

        return bboxes, labels

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        preprocessed_path = os.path.join(self.preprocessed_dir, img_name)

        if not os.path.exists(preprocessed_path):
            raise FileNotFoundError(f"Preprocessed file not found: {preprocessed_path}")

        # Load image tensor
        image = Image.open(preprocessed_path).convert("RGB")  # Convert RGBA to RGB

        if self.img_transform is not None:
            image = self.img_transform(image)
            
        # Load bounding boxes and labels
        if self.labels_dir:
            orig_bboxes, labels = self.load_labels(img_name)

            bboxes = torch.tensor(orig_bboxes, dtype=torch.float32)

            labels = torch.tensor(labels, dtype=torch.int64)

            # apply augmentations
            if self.augment:
                image, bboxes, labels = self.augmentations.apply(image, bboxes, labels)

            # Zoom augmentation: if after zooming no object (and bbox) is left: skip the image
            if bboxes.numel() == 0:
                return self.__getitem__((idx + 1) % len(self.image_names))

            return image, bboxes, labels
        else:
            return image


def get_dataloader(dataset, config) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader for the dataset
    """
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=config.batch_size,
                                             shuffle=config.shuffle,
                                             collate_fn=collate_fn)
    return dataloader


# source https://github.com/biyoml/PyTorch-SSD/blob/master/utils/data/dataloader.py

def collate_fn(batch):
    """
    Function to handle variable-sized data.
    """
    if isinstance(batch[0], tuple):  
        images, bboxes, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(bboxes), list(labels)
    else:
        images = torch.stack(batch, dim=0)
        return images

def plot_image_with_bboxes(image, bboxes_orig, labels, title="Image with Bounding Boxes"):
    img_height, img_width = image.shape[1], image.shape[2] 
    print(img_height, img_width)
    # Scale normalized bboxes to absolute pixel values for visualization
    # TODO: --> * 4 used for non flipped images: works
    # Issue with flipped ones
    # How to test: set flip prob to one and you will see :)
    # bboxes = bboxes_orig.copy()
    bboxes = copy.deepcopy(bboxes_orig)
    bboxes[:, [0, 2]] *= img_width
    bboxes[:, [1, 3]] *= img_height

    # Convert to integer values for plotting
    bboxes_abs = bboxes.to(torch.int)
    
    print("BBoxes for Visualization:", bboxes_abs)

    # Ensure labels are strings
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    labels = [str(l) for l in labels]

    # TODO: Image to RGB

    # Draw bboxes
    image_with_boxes = draw_bounding_boxes(image, bboxes_abs, labels=labels, colors="red", width=2)

    # image tensor to NumPy for visualization
    img = image_with_boxes.permute(1, 2, 0).numpy()

    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()