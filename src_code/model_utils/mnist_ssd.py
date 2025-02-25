import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import pandas as pd
import numpy as np
from .utils_mnist_ssd import cxcy_to_gcxgcy,  gcxgcy_to_cxcy, class_label
from .utils import cxcy_to_xy, find_IoU, xy_to_cxcy
import torch.nn.functional as F
import os
basic_conv = nn.Conv2d  # change this to 3d if needed


def pretty_print_module_list(module_list, x, print_net=False, max_colwidth=500):
    '''x: dummyinput [batch=1, C, H, W]'''
    pd.options.display.max_colwidth = max_colwidth
    df = pd.DataFrame({'Layer': list(map(str, module_list))})
    output_size = []
    for i, layer in enumerate(module_list):
        x = layer(x)
        output_size.append(tuple(x.size()))
    df['Output size'] = output_size
    if print_net:
        print('\n', df, '\n')
    return df['Output size'].tolist()

# ========================conv======================


class Conv(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, padding=1, stride=1, groups=1, spectral=False):
        super(Conv, self).__init__()
        if spectral:
            self.conv = spectral_norm(basic_conv(inC, outC, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups))
        else:
            self.conv = basic_conv(inC, outC, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups)

    def forward(self, x):
        x = self.conv(x)
        return x

# =======================norm======================


class No_norm(nn.Module):
    def __init__(self, c):
        super(No_norm, self).__init__()

    def forward(self, x):
        return x


def init_param(m):
    """
    Initialize convolution parameters.
    """
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class BaseConv(nn.Module):
    def __init__(self, conv_layers, input_size, chosen_fm=[-1, -2],
                 norm=nn.InstanceNorm2d, conv=Conv, act_fn=nn.LeakyReLU(0.01), spectral=False):
        '''
        conv_layers: list of channels from input to output
        for example, conv_layers=[1,10,20]
        --> module_list=[
            conv(1,10), norm, act_fn,
            conv(10,10), norm, act_fn,
            maxpool,
            conv(10, 20), norm, act_fn,
            conv(20, 20), norm, act_fn]
        '''
        super(BaseConv, self).__init__()
        # create module list
        self.module_list = nn.ModuleList()
        self.fm_id = []
        for i in range(len(conv_layers)-1):
            self.module_list.extend([
                conv(inC=conv_layers[i], outC=conv_layers[i+1], spectral=spectral),
                norm(conv_layers[i+1]),
                act_fn,
                conv(inC=conv_layers[i+1], outC=conv_layers[i+1], spectral=spectral),
                norm(conv_layers[i+1]),
                act_fn,
                nn.MaxPool2d(kernel_size=2)]
            )
            input_size = np.ceil(np.array(input_size) / 2)

            # select feature maps for prediction. They are the output of act_fn right before maxpool layers
            self.fm_id.append(len(self.module_list) - 2)

        self.fm_id = [self.fm_id[i] for i in chosen_fm]  # only use the last 2 fm in base conv

        self.module_list = self.module_list[:-1]  # ignore last maxpool layer

        self.output_size = input_size

    def forward(self, x):
        fm = []
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
            if i in self.fm_id:
                fm.append(x)
        return x, fm


class AuxConv(nn.Module):
    def __init__(self, conv_layers, input_size,
                 norm=nn.InstanceNorm2d, conv=Conv, act_fn=nn.LeakyReLU(0.01), spectral=False):
        '''
        conv_layers: list of channels from input to output.
        for example, conv_layers=[c1,c2,c3]
        --> module_list=[
            conv(c1, c1//2, kernel_size=1, pad=0), act_fn,
            conv(c1//2, c2, 3, 1, stride=2), act_fn,
            conv(c2, c2//2), norm, act_fn,
            conv(c2//2, c3), norm, act_fn]

        input_size: int
        '''
        super(AuxConv, self).__init__()
        self.module_list = nn.ModuleList()
        self.fm_id = []
        for i in range(len(conv_layers)-1):
            self.module_list.extend([conv(conv_layers[i], conv_layers[i]//2, kernel_size=1, padding=0, spectral=spectral),
                                     norm(conv_layers[i]//2),
                                     act_fn,
                                     conv(conv_layers[i]//2, conv_layers[i+1], kernel_size=3, padding=1, stride=2, spectral=spectral),
                                     norm(conv_layers[i+1]),
                                     act_fn])
            self.fm_id.append(len(self.module_list)-1)

    def forward(self, x):
        fm = []
        for i in range(len(self.module_list)):
            x = self.module_list[i](x)
            if i in self.fm_id:
                fm.append(x)
        return fm


class PredictionConv(nn.Module):
    def __init__(self, num_classes, fm_channels, n_prior_per_pixel, conv=Conv, norm=nn.InstanceNorm2d, spectral=False):
        super(PredictionConv, self).__init__()
        self.num_classes = num_classes

        # localization conv out layers
        self.loc_module_list = nn.ModuleList()
        for i in range(len(fm_channels)):
            self.loc_module_list.append(nn.Sequential(norm(fm_channels[i]),
                                        conv(fm_channels[i], n_prior_per_pixel[i]*4, kernel_size=3, padding=1, spectral=spectral)))

        # prediction conv out layers
        self.cla_module_list = nn.ModuleList()
        for i in range(len(fm_channels)):
            self.cla_module_list.append(nn.Sequential(norm(fm_channels[i]),
                                        conv(fm_channels[i], n_prior_per_pixel[i]*num_classes, kernel_size=3, padding=1, spectral=spectral)))

    def postprocess(self, x, k):
        '''x: output of self.(loc/cla)module_list. size [batch, n_boxes*k, h, w]. reshape into [batch, n_boxes*h*w, k]
           k: 4 or num_classes'''
        x = x.permute([0, 3, 2, 1]).contiguous()
        x = x.view(x.size(0), -1, k)
        return x

    def forward(self, fm):
        '''
        fm: list[n_fm] of torch tensors[batch, channel, h,w]: feature maps that contain priors
        return: loc_output[]
        '''
        loc_output = []
        cla_output = []

        for i in range(len(fm)):

            loc_output.append(self.postprocess(self.loc_module_list[i](fm[i]), 4))

            cla_output.append(self.postprocess(self.cla_module_list[i](fm[i]), self.num_classes))

        loc_output = torch.cat(loc_output, dim=1)  # [batch, total_n_prior, 4]
        cla_output = torch.cat(cla_output, dim=1)  # [batch, total_n_prior, num_classes]

        return loc_output, cla_output


class SSD(nn.Module):
    def __init__(self, configs, base_conv=None, aux_conv=None):
        super(SSD, self).__init__()
        self.configs = configs

        if base_conv != None:
            self.base_conv = base_conv
        else:
            self.base_conv = BaseConv(configs.base_conv_conv_layers, configs.base_conv_input_size, norm=No_norm)

        if aux_conv != None:
            self.aux_conv = aux_conv
        else:
            self.aux_conv = AuxConv(configs.aux_conv_conv_layers, configs.aux_conv_input_size, norm=No_norm)

        self.pred_conv = PredictionConv(configs.num_classes, configs.fm_channels, configs.n_prior_per_pixel)

        # prior boxes
        self.priors_cxcy = self.create_prior_boxes()
        self.n_p = len(self.priors_cxcy)

        self.apply(init_param)
        print('Done initialization')

    def forward(self, x):
        '''
        x: tensor[N, 3, 300, 300]
        returns predictions:
            loc_output (N, n_p, 4)
            cla_output (N, n_p, num_classes)
        '''
        x, fm = self.base_conv(x)

        fm.extend(self.aux_conv(x))

        loc_output, cla_output = self.pred_conv(fm)
        return loc_output, cla_output, fm

    def create_prior_boxes(self):
        '''
        input: configs.fm_size,
        'fm_prior_scale', 'fm_prior_aspect_ratio']

        return: prior boxes in center-size coordinates.
        Tensor size [n_p, 4]
        '''
        priors = []
        for i in range(self.configs.n_fm):
            d_h = self.configs.fm_size[i][0]
            d_w = self.configs.fm_size[i][1]
            s = self.configs.fm_prior_scale[i]
            h_w_ratio = d_h / d_w

            for j in range(d_w):
                for k in range(d_h):
                    # Note the order of k, j vs x,y here. It must be consistent with the permute/view operation in PredictionConv.post_process_output
                    cx = (j + 0.5)/d_w
                    cy = (k + 0.5)/d_h
                    for r in self.configs.fm_prior_aspect_ratio[i]:
                        w_scale = s * h_w_ratio
                        w_scale = (1/r) * w_scale
                        assert w_scale <= 1, f"{w_scale = }"
                        assert s <= 1, f"{w_scale = }"
                        priors.append([cx, cy, w_scale, s])
                        if r == 1:
                            additional_scale = 0.1
                            priors.append([cx, cy, additional_scale, additional_scale])
        priors = torch.FloatTensor(priors).to(self.configs.device)
        priors.clamp_(0, 1)
        print(f"There are {len(priors)} priors in this model")
        return priors

    def detect_object(self, loc_output, cla_output, min_score, max_overlap, top_k):
        '''
        loc_output: size [n, n_p, 4]
        cla_output: size [n, n_p, num_classes]

        return:
        '''
        # print('detecting...')
        batch_size = loc_output.size(0)

        cla_output = F.softmax(cla_output, dim=2)  # [N, N_P, num_classes]
        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(loc_output[i], self.priors_cxcy))  # (n_p, 4), fractional pt. coord
            image_boxes = []
            image_labels = []
            image_scores = []

            max_score, best_label = cla_output[i].max(dim=1)
            for c in range(self.configs.num_classes - 1):
                class_score = cla_output[i][:, c]  # [n_p]
                above_min_score_index = (class_score > min_score)  # [n_p]
                class_score = class_score[above_min_score_index]  # [n_p_min]
                if len(class_score) == 0:
                    continue
                sorted_score, sorted_index = class_score.sort(dim=0, descending=True)  # [n_p_min]

                keep = torch.ones_like(sorted_score, dtype=torch.uint8).to(self.configs.device)  # [n_p_min]
                iou = find_IoU(decoded_locs[above_min_score_index][sorted_index],
                                                 decoded_locs[above_min_score_index][sorted_index])  # [n_p_min, n_p_min]
                for j in range(len(sorted_index)-1):
                    if keep[j] == 1:
                        keep[j+1:] = torch.min(keep[j+1:], iou[j, j+1:] <= max_overlap)

                image_boxes.append(decoded_locs[above_min_score_index][sorted_index][keep])
                image_labels += [c] * keep.sum()
                image_scores.append(sorted_score[keep])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.configs.device))
                image_labels.append(torch.LongTensor([0]).to(self.configs.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.configs.device))

            image_boxes = torch.cat(image_boxes, dim=0)  # [n_detected_object_for_this_image, 4]
            image_labels = torch.tensor(image_labels)  # [n_detected_object_for_this_image, 1]
            image_scores = torch.cat(image_scores, dim=0)  # [n_detected_object_for_this_image, 1]

            assert len(image_boxes) == len(image_labels) == len(image_scores)
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
        return all_images_boxes, all_images_labels, all_images_scores
    
    def load_from_checkpoint(self):
        """ Loads model checkpoint if available."""
        # Ensure the filename is a correct absolute path
        chpt_path = self.configs.resume_from_checkpoint_path
        assert chpt_path is not None, "Please specify the checkpoint in the configs!"
        assert os.path.exists(chpt_path), f"{chpt_path} doesn't exist!"
        checkpoint = torch.load(chpt_path, weights_only=False, map_location=self.configs.device)
        
        self.load_state_dict(checkpoint["model_state"])
        # @todo, think about how we do the optimizer run 
        # self.optim.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded from {chpt_path} (starting from epoch {start_epoch})")

class MultiBoxLossSSD(nn.Module):
    def __init__(self, priors_cxcy, configs):
        super(MultiBoxLossSSD, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.n_p = self.priors_cxcy.size(0)
        self.configs = configs
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.loc_criterion = nn.L1Loss()
        self.cla_criterion = nn.CrossEntropyLoss(reduction='none')
        self.debug = configs.debug or configs.log_expt

    def initialise_debug_info(self):
        debug_info = {}
        debug_info["overlap_gt_def_boxes"] = []
        debug_info["db_for_each_obj"] = []
        debug_info["db_indices_for_each_obj"] = []
        debug_info["overlap_value_for_each_db"] = []
        debug_info["self.label_each_db"] = []
        debug_info["match"] = []
        debug_info["matched_gt_boxes"] = []
        debug_info["gt_locs"] = []
        debug_info["default_box_for_each_obj"] = []
        debug_info["pos_default_boxes"] = []
        debug_info["soft_maxed_pred"] = []
        return debug_info

    def forward(self, loc_output, cla_output, boxes, labels):
        '''
        loc_output: [N, N_P, 4]
        cla_output: [N, N_P, num_classes]
        boxes: list of N tensor, each tensor has size [N_objects, 4], frac. coord
        labels: list of N tensor, each tensor has size [N_objects]

        return loss: scalar
        '''
        loc_gt = torch.zeros_like(loc_output, dtype=torch.float)
        cla_gt = torch.zeros([len(boxes), self.n_p], dtype=torch.long).to(self.configs.device)
        boxes = [bb.to(self.configs.device) for bb in boxes]
        labels = [ll.to(self.configs.device) for ll in labels]
        if self.debug:
            debug_info = self.initialise_debug_info()
            debug_info["num_images"] = len(boxes)
        for i in range(len(boxes)):  # for each image in batch
            n_object = len(boxes[i])
            iou = find_IoU(boxes[i], self.priors_xy)  # (n_object, n_p)
            max_overlap_for_each_prior, object_for_each_prior = iou.max(dim=0)  # [n_p], [n_p]

            # make sure all gt boxes corresponds to at least one prior
            _, prior_for_each_object = iou.max(dim=1)  # [n_object]
            object_for_each_prior[prior_for_each_object] = torch.tensor(range(n_object)).to(self.configs.device)
            max_overlap_for_each_prior[prior_for_each_object] = 1.
            if self.debug:
                debug_info["db_for_each_obj"].append(prior_for_each_object)
                debug_info["matched_gt_boxes"].append(self.priors_xy[prior_for_each_object])
            loc_gt[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

            cla_gt[i] = labels[i][object_for_each_prior]
            cla_gt[i][max_overlap_for_each_prior < self.configs.pos_box_threshold] = class_label['background']
            if self.debug:
                pred_softmaxed = torch.nn.functional.softmax(cla_output[i], dim=1)[prior_for_each_object].to('cpu')
                debug_info["soft_maxed_pred"].append(pred_softmaxed)

        # get positives
        positives = (cla_gt != class_label['background'])  # [n, n_p]
        n_pos = positives.sum()
        # loc_loss
        self.loc_loss = self.loc_criterion(loc_output[positives], loc_gt[positives])  # scalar
        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # cla_loss, use hard_negative_mining on neg priors
        cla_loss = self.cla_criterion(cla_output.view(-1, self.configs.num_classes), cla_gt.view(-1))  # [N * n_p]
        cla_loss = cla_loss.view(-1, self.n_p)  # [N, n_p], so that we can use tensor positives for indexing

        cla_loss_pos = cla_loss[positives]
        cla_loss_neg = cla_loss[~positives].sort(dim=0, descending=True)[0][:int(n_pos*self.configs.neg_pos_hard_mining)]

        ce_pos_loss = cla_loss_pos.sum() / n_pos
        ce_neg_loss = cla_loss_neg.sum() / n_pos
        self.cla_loss = ce_pos_loss + ce_neg_loss  # self.configs.alpha * (cla_loss_pos.sum() + cla_loss_neg.sum()) / n_pos

        if self.debug:
            debug_info["ce_loss"] = self.cla_loss
            debug_info["loc_loss"] = self.loc_loss
            ce_hard_neg_loss = self.configs.alpha * (cla_loss_neg.sum()) / n_pos
            debug_info["ce_hard_neg_loss"] = ce_hard_neg_loss
            ce_pos_loss = self.configs.alpha * (cla_loss_pos.sum()) / n_pos
            debug_info["ce_pos_loss"] = ce_pos_loss
            debug_info["loss"] = self.loc_loss + self.cla_loss
        
        if not self.debug:
            debug_info = {}
        return self.configs.alpha * self.loc_loss + self.cla_loss, debug_info
