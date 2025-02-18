import torch
import torch.nn as nn
from src_code.model_utils.backbone import VGG16Backbone
import yaml
import os

# Load Configuration
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'configs_common.yaml'), 'r') as file:
    config = yaml.safe_load(file)

NUM_CLASSES = config['task_configs']['num_classes']
NUM_DEFAULT_BOXES = 4  # Number of anchors per grid cell

class SSD300(nn.Module):
    """SSD model producing pred_locs and cls_locs for each default box."""
    def __init__(self, num_classes=NUM_CLASSES, num_default_boxes=NUM_DEFAULT_BOXES):
        super().__init__()
        self.backbone = VGG16Backbone(pretrained=True)
        
        # Prediction heads
        self.loc_head = nn.Conv2d(1024, num_default_boxes * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(1024, num_default_boxes * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Get feature maps from backbone
        conv4_3, conv7 = self.backbone(x)
        
        # Predict locations and classes
        pred_locs = self.loc_head(conv7)
        pred_cls = self.cls_head(conv7)
        
        # Reshape outputs for each default box
        pred_locs = pred_locs.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, NUM_CLASSES)
        
        return pred_locs, pred_cls

# Test the Model
'''
if __name__ == "__main__":
    model = SSD300()
    test_input = torch.randn(1, 3, 300, 300)
    pred_locs, pred_cls = model(test_input)
    print(f"Localization Output Shape (pred_locs): {pred_locs.shape}")
    print(f"Classification Output Shape (pred_cls): {pred_cls.shape}")
'''