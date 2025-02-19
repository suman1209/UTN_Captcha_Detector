from torch import nn
import torch
from .utils import cxcy_to_torch_xy, torch_xy_to_cxcy, encode_bboxes, find_IoU
from src_code.data_utils.dataset_utils import label_map
from src_code.task_utils.config_parser import ConfigParser


class MultiBoxLoss(nn.Module):
    def __init__(self, default_boxes: list[torch.tensor],
                 config: ConfigParser):
        """
        Parameters
        ----------
        default_boxes(List[torch.tensor]): the prior boxes
        config.pos_box_threshold(float): threshold to determine pos boxes
        config.neg_pos_hard_mining(int): for hard negative mining
        config.alpha: loss = alpha * loc_loss + ce_loss
        """
        super().__init__()
        self.db = default_boxes
        self.threshold = config.pos_box_threshold
        self.debug = config.debug
        self.neg_pos = config.neg_pos_hard_mining
        self.alpha = config.alpha
        h, w = config.img_height, config.img_height
        self.img_scale = torch.tensor([h, w, h, w])
        self.ds_factor = config.downscale_factor

    def __verify__(self, threshold):
        msg = f"{threshold = } of type type {threshold}"
        assert isinstance(threshold, float), msg

    def initialise_debug_info(self):
        debug_info = {}
        debug_info["overlap_gt_def_boxes"] = []
        debug_info["db_for_each_obj"] = []
        debug_info["db_indices_for_each_obj"] = []
        debug_info["overlap_value_for_each_db"] = []
        debug_info["self.label_each_db"] = []
        debug_info["match"] = []
        debug_info["gt_locs"] = []
        return debug_info

    def forward(self, locs_pred, cls_pred, boxes, labels, downscale_factor=4):
        """
        Parameters
        ----------
        locs_pred(torch.tensor): (N_batch, N_db, 4)
        cls_pred(torch.tensor):  (N_batch, N_db, n_classes=36)
        boxes(List[torch.tensor]): True obj bouding boxes, a list of N tensors
        labels(List[torch.tensor]): True obj labels, a list of N tensors
        downscale_factor(float): downscale factor used in the preprocessing
        Returns
        -------
        loss(torch.float32): Mutilbox loss
        debug_info(dict): contains intermediate info about the loss calculation
        """
        dev = locs_pred.device
        labels = [label.to(dev) for label in labels]
        boxes = [box.to(dev) for box in boxes]
        if self.debug:
            debug_info = self.initialise_debug_info()
            debug_info["num_images"] = len(boxes)
        batch_size = locs_pred.size(0)  # N_batch
        n_db = self.db.size(0)  # N_def_boxes
        num_classes = cls_pred.size(2)  # N_classes

        # initialise gt loc offsets and gt labels
        # (N_batch, N_def_boxes, 4)
        gt_locs = torch.zeros((batch_size, n_db, 4), dtype=torch.float).to(dev)
        # (N_batch, N_def_boxes)
        gt_classes = torch.zeros((batch_size, n_db), dtype=torch.long).to(dev)
        img_scale = self.img_scale.unsqueeze(0)
        scaled_default_boxes = self.db.to(dev) * img_scale / self.ds_factor
        default_boxes_xy = cxcy_to_torch_xy(scaled_default_boxes)
        loss = -1

        """
        populate the ground-truth loc offsets and labels
        for each default boxes in eahc image of the batch
        """
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            # (N_objects, N_def_boxes)
            overlap = find_IoU(boxes[i], default_boxes_xy)
            if self.debug:
                debug_info["overlap_gt_def_boxes"].append(overlap)
            # (N_def_boxes)
            overlap_value_for_each_db, obj_for_each_db = overlap.max(dim=0)
            if self.debug:
                data = overlap_value_for_each_db
                debug_info["overlap_value_for_each_db"] = data
            _, db_for_each_obj = overlap.max(dim=1)  # (N_objects)
            if self.debug:
                debug_info["db_for_each_obj"].append(db_for_each_obj)
            obj_for_each_db = obj_for_each_db.to(dev)
            obj_for_each_db_copy = obj_for_each_db.clone()
            db_for_each_obj = db_for_each_obj.to(dev)
            if self.debug:
                debug_info["db_indices_for_each_obj"].append(db_for_each_obj)
            # assign each object to the corresponding maximum-overlap-prior
            obj_indices = torch.LongTensor(range(n_objects)).to(dev)
            obj_for_each_db[db_for_each_obj] = obj_indices
            msg = f"{obj_for_each_db_copy}!={obj_for_each_db}"
            # assert torch.equal(obj_for_each_db_copy, obj_for_each_db), msg
            # artificially elevate the matching scores for the matching boxes
            overlap_value_for_each_db[db_for_each_obj] = 1.
            # each db gets the label of the object to which it is matched
            self.label_each_db = labels[i][obj_for_each_db]  # (N_def_boxes)
            if self.debug:
                match_mask = overlap_value_for_each_db >= self.threshold
                debug_info["match"].append(match_mask)
            thresh_mask = overlap_value_for_each_db < self.threshold
            # (N_def_boxes)
            self.label_each_db[thresh_mask] = label_map["background"]
            if self.debug:
                debug_info["self.label_each_db"].append(self.label_each_db)
            # Save gt_classes
            gt_classes[i] = self.label_each_db

            # Encode pred bboxes (finding the ground-truth offsets)
            gt_xy_for_each_db = boxes[i][obj_for_each_db] * self.ds_factor
            gt_cxcy_for_each_default_box = torch_xy_to_cxcy(gt_xy_for_each_db)
            # remember that we need to scale it back to normalised coordinates
            gt_cxcy_for_each_default_box /= self.img_scale
            # (N_def_boxes, 4)
            gt_locs[i] = encode_bboxes(gt_cxcy_for_each_default_box, self.db)
            if self.debug:
                debug_info["gt_locs"].append(gt_locs[i])
        # 1. Localization loss
        # Identify priors that are positive
        # bool Tensor mask (N_batch, N_def_boxes)
        # pos_db -> positive default boxes
        pos_db = gt_classes != label_map["background"]
        if self.debug:
            debug_info["gt_locs"].append(gt_locs[i])
        # Localization loss is computed only over positive default boxes

        smooth_L1_loss = nn.SmoothL1Loss()
        loc_loss = smooth_L1_loss(locs_pred[pos_db], gt_locs[pos_db])
        if self.debug:
            debug_info["loc_loss"] = loc_loss

        # 2. Confidence loss
        # Apply hard negative mining

        # number of positive and hard-negative default boxes per image
        n_positive = pos_db.sum(dim=1)
        n_hard_negatives = self.neg_pos * n_positive
        if self.debug:
            debug_info["n_positive"] = n_positive
            debug_info["n_hard_negatives"] = n_hard_negatives
        # Find the loss for all priors
        cross_entropy_loss = nn.CrossEntropyLoss(reduce=False)
        pred = cls_pred.view(-1, num_classes)
        gt = gt_classes.view(-1)
        if self.debug:
            debug_info["gt_label_each_default_box"] = gt
        conf_loss_all = cross_entropy_loss(pred, gt)  # (N_batch*N_def_boxes)
        # (N_batch, N_def_boxes)
        conf_loss_all = conf_loss_all.view(batch_size, n_db)
        if self.debug:
            debug_info["conf_loss_for_each_default_box"] = conf_loss_all
        confidence_pos_loss = conf_loss_all[pos_db]
        if self.debug:
            debug_info["confidence_pos_loss"] = conf_loss_all
        # Find which priors are hard-negative
        conf_neg_loss = conf_loss_all.clone()  # (N_batch, N_def_boxes)
        conf_neg_loss[pos_db] = 0.
        conf_neg_loss, _ = conf_neg_loss.sort(dim=1, descending=True)

        """
        the following performs hard-negative mining, in each image of a batch,
        there will be different number of positive boxes.
        """
        def_box_indices = torch.LongTensor(range(n_db)).unsqueeze(0)
        # (N_batch, N_def_boxes)
        hardness_ranks = def_box_indices.expand_as(conf_neg_loss).to(dev)
        # mask(N_batch, N_def_boxes)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        if self.debug:
            debug_info["hard_negatives"] = hard_negatives
        num_pos_boxes = n_positive.sum().float()
        confidence_hard_neg_loss = conf_neg_loss[hard_negatives]
        total_sum = confidence_hard_neg_loss.sum() + confidence_pos_loss.sum()
        confidence_loss = total_sum / num_pos_boxes
        loss = self.alpha * loc_loss + confidence_loss
        if self.debug:
            debug_info["conf_neg_loss"] = conf_loss_all
            debug_info["ce_loss"] = confidence_loss
            debug_info["loc_loss"] = loc_loss
            debug_info["ce_loss"] = confidence_loss
            ce_hard_neg_loss = confidence_hard_neg_loss.sum() / num_pos_boxes
            debug_info["ce_hard_neg_loss"] = ce_hard_neg_loss
            ce_pos_loss = confidence_pos_loss.sum() / num_pos_boxes
            debug_info["ce_pos_loss"] = ce_pos_loss
            debug_info["loss"] = loss
        if debug_info is None:
           debug_info = {} 
        return loss, debug_info
