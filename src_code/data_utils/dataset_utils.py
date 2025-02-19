import torch
import os
from torchvision.datasets import VisionDataset
from .augmentation import Augmentations

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
    def __init__(self, config):
        """
        Captcha Dataset for loading preprocessed data

        preprocessed_dir: preprocessed tensors
        labels_dir: label files
        """
        super().__init__(config.preprocessed_dir)

        self.preprocessed_dir = config.preprocessed_dir
        self.labels_dir = config.labels_dir
        self.augment = config.augment
        self.augmentations = Augmentations(config)

        self.image_names = sorted([f for f in os.listdir(config.preprocessed_dir) if f.endswith(".pt")])

    def load_labels(self, image_name):
        """
        Loading bounding boxes and class labels
        """

        label_file = os.path.join(self.labels_dir, image_name.replace(".pt",
                                                                      ".txt"))

        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        bboxes = []
        labels = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                class_label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2

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
        image = torch.load(preprocessed_path)

        # Load bounding boxes and labels
        orig_bboxes, labels = self.load_labels(img_name)

        bboxes = torch.tensor(orig_bboxes, dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64)

        # apply augmentations
        if self.augment:
            image, bboxes = self.augmentations.apply(image, bboxes)

        return image, bboxes, labels


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
    images, bboxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(bboxes), list(labels)
