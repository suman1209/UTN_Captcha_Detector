import torch
import torch.nn as nn
from .backbone import VGG16Backbone
import yaml
import os
import torch.nn.functional as F

# Load Configuration
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'configs_common_simple.yaml'), 'r') as file:
    config = yaml.safe_load(file)

NUM_CLASSES = config['task_configs']['num_classes']
# Number of anchors per grid cell
NUM_DEFAULT_BOXES_PER_PIXEL = {'conv4_3': 4,
                               'conv7': 4,
                               'conv8_2': 4,
                               'conv9_2': 4}


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out))
        conv8_2_feats = out

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_2_feats = out

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats


class PredictionHead(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes, n_boxes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionHead, self).__init__()

        self.n_classes = n_classes

        # 4 prior-boxes implies we use 4 different aspect ratios, etc.
        print(f"{n_boxes = }")
        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats):
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3,
                                      1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1,
                                   self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1,
                               self.n_classes)

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2], dim=1)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2], dim=1)

        return locs, classes_scores


class SSDCaptcha(nn.Module):
    """
    SSD model with auxiliary network and default box generation.
    """
    def __init__(self, num_classes=NUM_CLASSES, n_boxes_per_pixel=NUM_DEFAULT_BOXES_PER_PIXEL, checkpoint_path=None):
        # checkpoitnt_path = "./src_code/model_utils/vgg_counter_checkpoint_2025-02-22_01-54-43.pth"
        super().__init__()
        self.num_classes = num_classes
        self.n_boxes_per_pixel = n_boxes_per_pixel

        # Backbone Network (VGG16-based feature extractor)
        # self.backbone = VGG16Backbone(pretrained=False)
        self.backbone = self.load_backbone(checkpoint_path)

        # Auxiliary Network (extra feature maps for detecting smaller objects)
        self.auxiliary_convs = AuxiliaryConvolutions()

        # Prediction heads (for bounding box locations and class scores)
        self.prediction_head = PredictionHead(NUM_CLASSES, n_boxes_per_pixel)
        self.fm_info = {}

    def load_backbone(self, checkpoint_path: str):
        # custom pretrained model by dhimitri
        if checkpoint_path is not None:
            backbone = VGG16Backbone(pretrained=False)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            backbone.load_state_dict(checkpoint)
        else:
            backbone = VGG16Backbone(pretrained=False)
        return backbone

    def forward(self, x):
        """
        Forward pass of SSD model.
        :param x: Input image tensor of shape (batch_size, 3, height, width)
        :return: Tuple (predicted bounding box locations, predicted class scores)
        """
        # Extract base feature maps from VGG16 backbone
        _,  conv4_3_feats, conv7_feats = self.backbone(x)
        # print(f"{conv4_3_feats.shape = },  {conv7_feats.shape = }")
        # assert False
        conv8_2_feats, conv9_2_feats = self.auxiliary_convs(conv7_feats)

        self.fm_info['conv4_3'] = list(conv4_3_feats.shape[-2:])
        self.fm_info['conv7'] = list(conv7_feats.shape[-2:])
        self.fm_info['conv8_2_feats'] = list(conv8_2_feats.shape[-2:])
        self.fm_info['conv9_2_feats'] = list(conv9_2_feats.shape[-2:])
        # self.fm_info['conv10_2_feats'] = list(conv10_2_feats.shape[-2:])
        # self.fm_info['conv11_2_feats'] = list(conv11_2_feats.shape[-2:])
        # generate the predicted location offsets and classes
        pred_locs, pred_cls = self.prediction_head(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats)

        return pred_locs, pred_cls, self.fm_info

    def generate_default_boxes(self, fmap_hw={'conv4_3': [20, 80], 'conv7': [10, 40], 'conv8_2_feats': [5, 20], 'conv9_2_feats': [3, 10]}):
        '''
            Create 3610 default boxes in center-coordinate,
            a tensor of dimensions (num_default_boxes, 4)
        '''
        device = "cuda"

        scales = {"conv4_3": 0.2, "conv7": 0.4, "conv8_2_feats": 0.7, "conv9_2_feats": 0.9}

        # ratio = h/w
        aspect_ratios = {"conv4_3": [1., 2, 3, 0.5], "conv7": [1., 2, 3, 0.5], "conv8_2_feats": [1., 2, 3, 0.5], "conv9_2_feats": [1., 2, 3, 0.5]}

        fmaps = list(fmap_hw.keys())

        default_boxes = []

        for k, fmap in enumerate(fmaps):
            fm_height = fmap_hw[fmap][0]
            fm_width = fmap_hw[fmap][1]
            h_w_ratio = fm_height / fm_width
            for ratio in aspect_ratios[fmap]:
                for i in range(fmap_hw[fmap][0]):
                    for j in range(fmap_hw[fmap][1]):
                        cx = (i + 0.5) / fm_height
                        cy = (j + 0.5) / fm_width
                        # (cx, cy, w, h)
                        h_scale = scales[fmap]
                        # ratio = h/w
                        # breakpoint()
                        w_scale = h_scale * h_w_ratio
                        w_scale = (1/ratio) * w_scale
                        default_boxes.append([cy, cx, w_scale, h_scale])

        default_boxes = torch.FloatTensor(default_boxes).to(device)  # (8732, 4)
        default_boxes.clamp_(0, 1)
        # if not self.debug:
        #     assert default_boxes.size(0) == self.total_box_count, f"got {len(default_boxes)}, expected {self.total_box_count} boxes"
        assert default_boxes.size(1) == 4
        return default_boxes

# Example usage:


if __name__ == "__main__":
    model = SSDCaptcha()
    test_input = torch.randn(1, 1, 40, 160)  # Maintain rectangular aspect ratio
    pred_locs, pred_cls, fm_info = model(test_input)
    print(f"Localization Output Shape: {pred_locs.shape}")
    print(f"Classification Output Shape: {pred_cls.shape}")
    print(f"fm_info: {fm_info = }")
    # Generating default boxes for different feature maps
    feature_map_shapes = fm_info.values()  # Example feature map sizes for rectangular input
    default_boxes = model.generate_default_boxes(feature_map_shapes)
    print(f"Generated {default_boxes.shape[0]} default boxes.")
