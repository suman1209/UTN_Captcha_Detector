import torch
import torch.nn as nn
from src_code.model_utils.backbone import VGG16Backbone
import yaml
import os

# Load Configuration
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'configs_common.yaml'), 'r') as file:
    config = yaml.safe_load(file)

NUM_CLASSES = config['task_configs']['num_classes']
NUM_DEFAULT_BOXES_PER_PIXEL = 4  # Number of anchors per grid cell

class SSD(nn.Module):
    """
    SSD model with auxiliary network and default box generation.
    """
    def __init__(self, num_classes=NUM_CLASSES, num_default_boxes=NUM_DEFAULT_BOXES_PER_PIXEL):
        super().__init__()
        self.num_classes = num_classes
        self.num_default_boxes = num_default_boxes
        
        # Backbone Network (VGG16-based feature extractor)
        self.backbone = VGG16Backbone(pretrained=True)
        
        # Auxiliary Network (extra feature maps for detecting smaller objects)
        self.auxiliary_convs = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),  # Reduce channels to 256
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Downsample spatial dimensions
            nn.ReLU(),
            
            nn.Conv2d(512, 128, kernel_size=1),  # Further channel reduction
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Another downsampling
            nn.ReLU(),
        )
        
        # Prediction heads (for bounding box locations and class scores)
        self.loc_head = nn.Conv2d(1024, num_default_boxes * 4, kernel_size=3, padding=1)  # Localization predictions
        self.cls_head = nn.Conv2d(1024, num_default_boxes * num_classes, kernel_size=3, padding=1)  # Class predictions
        
    def forward(self, x):
        """
        Forward pass of SSD model.
        :param x: Input image tensor of shape (batch_size, 3, height, width)
        :return: Tuple (predicted bounding box locations, predicted class scores)
        """
        # Extract base feature maps from VGG16 backbone
        conv4_3, conv7 = self.backbone(x)
        
        # Predict bounding box locations and class scores using the main feature maps (conv7)
        pred_locs = self.loc_head(conv7)  # Bounding box offsets
        pred_cls = self.cls_head(conv7)  # Class confidence scores
        
        # Reshape outputs to match the number of default boxes
        pred_locs = pred_locs.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)  # (batch_size, num_boxes, 4)
        pred_cls = pred_cls.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)  # (batch_size, num_boxes, num_classes)
        
        return pred_locs, pred_cls
    
    def generate_default_boxes(self, feature_map_shapes, aspect_ratios=[1.0, 2.0, 0.5]):
        """
        Generate default anchor boxes for each feature map.
        :param feature_map_shapes: List of tuples (height, width) representing each feature map's dimensions.
        :param aspect_ratios: List of aspect ratios to use for default boxes.
        :return: Tensor of shape (num_default_boxes, 4) with box coordinates.
        """
        default_boxes = []
        
        for feature_shape in feature_map_shapes:
            f_h, f_w = feature_shape  # Feature map height and width
            for i in range(f_h):
                for j in range(f_w):
                    for ratio in aspect_ratios:
                        # Center coordinates of the default box (normalized to [0,1])
                        cx = (j + 0.5) / f_w
                        cy = (i + 0.5) / f_h
                        
                        # Compute width and height of the default box based on aspect ratio
                        w = 1.0 / f_w * ratio
                        h = 1.0 / f_h / ratio
                        
                        default_boxes.append([cx, cy, w, h])
        
        return torch.tensor(default_boxes, dtype=torch.float32)  # Return as a PyTorch tensor

# Example usage:
'''
if __name__ == "__main__":
    model = SSD()
    test_input = torch.randn(1, 3, 640, 160)  # Maintain rectangular aspect ratio
    pred_locs, pred_cls, aux_features = model(test_input)
    print(f"Localization Output Shape: {pred_locs.shape}")
    print(f"Classification Output Shape: {pred_cls.shape}")
    
    # Generating default boxes for different feature maps
    feature_map_shapes = [(40, 10), (20, 5)]  # Example feature map sizes for rectangular input
    default_boxes = model.generate_default_boxes(feature_map_shapes)
    print(f"Generated {default_boxes.shape[0]} default boxes.")
'''