from backbone import VGG16Backbone
import torch
import torch.nn as nn
from torchvision import models
import torchvision
import torch.nn.functional as F
import sys
sys.path.insert(0, "../../")
from src_code.data_utils.dataset_utils import CaptchaDataset, get_dataloader
from src_code.task_utils.config_parser import ConfigParser

class CountBackbone(nn.Module):

    # backbone using VGG for feature extraction and predicting char count

    def __init__(self):
        super(CountBackbone, self).__init__()
        self.vgg_backbone = VGG16Backbone(pretrained=False)  # Load VGG16
        self.fc = nn.Linear(1024, 1)  # Final layer for regression


    def forward(self, x):
        # skipping conv4_3_feats and ony extract conv7
        out, conv2_2_feats, conv3_3_feats, = self.vgg_backbone(x)  
        x = F.adaptive_avg_pool2d(out, (1, 1))  # Global average pooling
        x = torch.flatten(x, 1)
        print(f"{x.shape =}")
        x = self.fc(x)  # Fully connected layer
        x = F.relu(x)
        return x




configs_dict = {
    "data_configs": {
        "train_path": "../datasets/utn_dataset_curated/part2/train",
        "val_path": "../datasets/utn_dataset_curated/part2/val",
        "test_path": "../datasets/utn_dataset_curated/part2/test",
        "preprocessing_related": {
            "mean": 0.5,  # for raw_image normalisation
            "std": 0.5,  # for raw_image normalisation
            "downscale_factor": 4,
        },
        "dataset_related": {
            "preprocessed_dir": "/var/lit2425/jenga/suman/pjf/computer_vision/UTN_Captcha_Detector/datasets/utn_dataset_curated/part2/train/preprocessed",
            "labels_dir": "/var/lit2425/jenga/suman/pjf/computer_vision/UTN_Captcha_Detector/datasets/utn_dataset_curated/part2/train/labels",
            "augment": True,
            "shuffle": False,
        },
        "augmentation_related": {
            "flip_prob": 0.5,
            "scale_range": (0.8, 1.2),
            "zoom_prob": 0.3,
            "saturation_prob": 0.2
        },
    },
    "model_configs": {
        "epochs": 1,
        "batch_size": 32,
        "device": "cuda",  # either "cpu" or "cuda"
        "checkpoint": None,
        "backbone": {
            "name": "VGG16",
            "num_stages": 6,
        },
        "loss": {
            "alpha": 1,  # loss = alpha*loc_loss + cls_loss
            "pos_box_threshold": 0.5,  # a default box is marked positive if it has (> pos_box_threshold) IoU score with any of the groundtruth boxes
            "hard_neg_pos": 3,  # num of negative boxes = hard_neg_pos * num_positive_boxes
        },
        "optim": {
            "name": "SGD",
            "lr": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
        },
        "scheduler": {
            "name": "MultiStepLR",
            "milestones": [155, 195],
            "gamma": 0.1,
        },
    },
    "task_configs": {
        "img_height": 160,  # original image height
        "img_width": 640,  # original image width
        "debug": True,  # if True will display a lot of intermediate information for debugging purposes
        "log_expt": False,  # whether to log the experiment online or not
        "num_classes": 37,  # A-Z(26), 0-9(10), background(1)
        "min_cls_score": 0.01,  # if the cls score for a bounding box is less than this, it is considered as background
        "nms_iou_score": 0.1,  # if the iou between two bounding boxes is less than this, it is suppressed
    },
}

# # hyperparameters
preprocessed_dir = "../../../datasets/utn_dataset_curated/part2/train/preprocessed"
labels_dir = "../datasets/utn_dataset_curated/part2/train/labels"
config = ConfigParser(configs_dict).get_parser()

train_set = CaptchaDataset(config)
image, bboxes, labels = train_set[0]


train_loader = get_dataloader(train_set, config)
images, bboxes, labels = next(iter(train_loader))
print(f"Batch Image Shape: {images.shape}")  # Expected: (batch_size, 1, 40, 160)

model = CountBackbone()

output = model(images)

print(f"Model Output Shape: {output}")  