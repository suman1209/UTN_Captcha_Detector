import math
import time
import random
from torch import nn, optim
import torch.utils.data
from torch.utils.data import DataLoader
from src_code.model_utils.loss import MultiBoxLoss
from src_code.model_utils.ssd import SSDCaptcha
from src_code.data_utils.preprocessing import *
from src_code.task_utils.config_parser import ConfigParser
from src_code.data_utils.dataset_utils import category_id_labels
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
import os
import numpy as np
from src_code.model_utils import utils_mnist_ssd
from src_code.model_utils.mnist_ssd import SSD, BaseConv, pretty_print_module_list, AuxConv, MultiBoxLossSSD
import torch
from tqdm import tqdm
from pathlib import Path as p
from datetime import datetime
from src_code.task_utils.evaluation import levenshtein
from src_code.task_utils.evaluation import generate_edit_distance
os.environ['WANDB_CACHE_DIR'] = "./"
os.environ['WANDB_DATA_DIR'] = "./"


class Metrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CaptchaTrainer:

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 loss_fn: nn.Module,
                 optimizer: optim,
                 config: ConfigParser,
                 logger=None, checkpoint_path="checkpoints"
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optim = optimizer
        self.config = config
        self.logger = logger
        self.start_epoch = 0  # Default start epoch
        self.checkpoint_path = checkpoint_path
        self.prev_map = -1
        self.prev_edit_distance = -1
        self.map = -2
        self.edit_distance = -2
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_path, exist_ok=True)
    def save_chpt_condition(self):
        # save every 10% progress
        # percent = 10
        # save_range = range(0, self.config.epochs, math.ceil(self.config.epochs/percent))
        # save_epoch_list = list(save_range)
        # epoch in save_epoch_list and self.save_checkpoint
        if self.edit_dist > self.prev_edit_distance:
            return True
        return False

    def train_step(self, epoch):
        self.model.to(self.config.device)
        losses = Metrics()
        ce_losses = Metrics()
        loc_losses = Metrics()
        ce_pos_losses = Metrics()
        ce_neg_losses = Metrics()
        assert len(self.train_loader) > 0, f"{len(self.train_loader) = }"
        num_batches = len(self.train_loader)
        for i, (images, boxes, labels) in enumerate(self.train_loader):
            images = images.to(self.config.device)  # (batch_size (N), 3, 160, 640)
            images.requires_grad = True
            self.optim.zero_grad()
            # Foward pass
            loc_pred, cls_pred, fm_info = self.model(images)
            # loss
            loss, debug_info = self.loss_fn(loc_pred, cls_pred, boxes, labels)
            # Backward pass
            loss.backward()

            self.optim.step()
            # Extract and detach loss components
            ce_loss = debug_info.get('ce_loss', torch.tensor(0.0, device=self.config.device)).detach().cpu().item()
            loc_loss = debug_info.get('loc_loss', torch.tensor(0.0, device=self.config.device)).detach().cpu().item()
            ce_pos_loss = debug_info.get('ce_pos_loss', torch.tensor(0.0, device=self.config.device)).detach().cpu().item()
            ce_neg_loss = debug_info.get('ce_hard_neg_loss', torch.tensor(0.0, device=self.config.device)).detach().cpu().item()
            loss_value = loss.detach().cpu().item()
            # Update metrics
            losses.update(loss_value, images.size(0))
            ce_losses.update(ce_loss, images.size(0))
            loc_losses.update(loc_loss, images.size(0))
            ce_pos_losses.update(ce_pos_loss, images.size(0))
            ce_neg_losses.update(ce_neg_loss, images.size(0))

            if self.logger is not None:
                self.logger.log({
                    "train_loss": loss_value,
                    "ce_loss": ce_loss,
                    "loc_loss": loc_loss,
                    "ce_pos_loss": ce_pos_loss,
                    "ce_neg_loss": ce_neg_loss
                })

            if i % self.config.print_freq == 0:
                print(f"Epoch: {epoch}({i}/{num_batches}) | Loss: {losses.avg:.4f} | CE Loss: {ce_losses.avg:.4f} | "
                      f"Loc Loss: {loc_losses.avg:.4f} | CE Pos Loss: {ce_pos_losses.avg:.4f} | "
                      f"CE Neg Loss: {ce_neg_losses.avg:.4f}")

            del loc_pred, cls_pred, images, boxes, labels, debug_info
            torch.cuda.empty_cache()
        return losses, ce_losses, loc_losses, ce_pos_losses, ce_neg_losses

    def validation_step(self, epoch):
        # rand image and draw the ground truth bbox and the matched default box
        random_image = random.randint(0, self.config.batch_size - 1)
        self.model.eval()
        self.model.to(self.config.device)
        only_once = False
        all_boxes_output = []
        all_labels_output = []
        all_scores_output = []
        all_boxes_gt = []
        all_labels_gt = []
        all_difficulties_gt = []

        self.model.eval()
        with torch.no_grad():
            for images, boxes, labels in tqdm(self.val_loader):  # labels: list[n] of tensors[n_object]

                loc_output, cla_output, _ = self.model(images.to(self.config.device))  # loc_output: tensor[n,n_p,4], cla_output: tensor[n, n_p, n_classes]
                boxes_output, labels_output, scores_output = self.model.detect_object(loc_output, cla_output,
                 min_score=self.config.nms_min_cls_score,
                 max_overlap=self.config.nms_iou_score,
                 top_k=self.config.nms_topk)

                all_boxes_output.extend(boxes_output)
                all_labels_output.extend(labels_output)
                all_scores_output.extend(scores_output)

                all_boxes_gt.extend(boxes)
                all_labels_gt.extend(labels)
                all_difficulties_gt = [torch.zeros_like(i, dtype=torch.bool) for i in all_labels_gt]
        APs, mAP = utils_mnist_ssd.calculate_mAP(all_boxes_output, all_labels_output, all_scores_output, all_boxes_gt, all_labels_gt, all_difficulties_gt)
        edit_distance, count = generate_edit_distance(self.model, self.val_loader, self.config)
        if self.config.debug:
            print(f"{APs = }")
        print(f"{mAP = }")
        print(f"{edit_distance = }")
        self.map = mAP
        self.edit_distance = edit_distance
        if self.config.log_expt:
            self.logger.log({'mAP': mAP})
            self.logger.log({'edit_distance': edit_distance})
        captcha_max_len = 10
        predicted_captcha = "".join([category_id_labels[i.item()] for i in all_labels_output[random_image]][:captcha_max_len])
        predicted_boxes = None
        matched_boxes = None
        if self.config.log_expt:
            with torch.no_grad():
                for i, (images, boxes, labels_gt) in enumerate(self.val_loader):
                    images = images.to(self.config.device)
                    loc_pred, cls_pred, fm_info = self.model(images)
                    loss, debug_info = self.loss_fn(loc_pred, cls_pred, boxes, labels_gt)
                    predicted_boxes, labels, scores = self.model.detect_object(loc_pred, cls_pred, min_score=0.25, max_overlap=0.5, top_k=20)
                    
                    list_boxes = predicted_boxes[random_image].tolist()
                    assert len(list_boxes) == len(labels[random_image])
                    for i, label_idx in enumerate(labels[random_image]):
                        list_boxes[i].append(label_idx.item())
                    list_boxes = sorted(list_boxes, key=lambda x: x[0])
                    
                    predicted_captcha = "".join([self.config.category_id_labels[i[-1]] for i in list_boxes])
                    str_labels = ["".join([category_id_labels[i.item()] for i in label]) for label in labels_gt]
                    edit_distance = levenshtein(str_labels[random_image], predicted_captcha)
                    predicted_boxes = list_boxes
                    if debug_info == {}:
                        break
                    
                    if not only_once:

                        img_np = images[random_image].cpu().detach().numpy().transpose(1, 2, 0)
                        label = str_labels[random_image]
                        gt_boxes = boxes[random_image].cpu().numpy()
                        neg_boxes = None
                        # if match_boxes is None:
                        #     pb = -1
                        # else:
                        #     pb = len(matched_boxes)
                        self.plot_bb(img_np, gt_boxes, matched_boxes, neg_boxes, predicted_boxes, f"epoch={epoch} label = {label} {predicted_captcha = } {edit_distance = }", i)
                        # logits = debug_info["soft_maxed_pred"][random_image]
                        # GT_int = labels[random_image].tolist()
                        # GT_str = str_labels[random_image]
                        # my_table = wandb.Table(columns=["GT"] + list(category_id_labels.values()) + ["bg"], data=[[f"{GT_str[i]}-{GT_int[i]}"] + logit.tolist() for i, logit in enumerate(logits)])

                        # self.log_logits(my_table, epoch)
                    break

    def log_logits(self, my_table, epoch):
        columns = my_table.columns
        data = my_table.data

        # Determine the number of rows and columns for the grid
        num_rows = len(data)
        num_cols = 1  # One column for each row's bar plot
        grid_size = (num_rows, num_cols)

        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))
        fig.suptitle(f"Bar Plots for Each Row {epoch = }", fontsize=16)

        # If there's only one row, axes will not be an array, so we wrap it in a list
        if num_rows == 1:
            axes = [axes]

        # Plot bar plots for each row
        for row_idx, row in enumerate(data):
            # Extract the row data (skip the first column, which is "GT")
            row_values = [float(x) for x in row[1:]]
            row_labels = columns[1:]  # Skip the "GT" column

            # Create a bar plot for the current row
            ax = axes[row_idx]
            ax.bar(row_labels, row_values)
            ax.set_title(f"Row {row_idx + 1} ({row[0]})")  # Use the "GT" value as the title
            ax.set_ylabel("Value")
            ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle

        # Log the figure to wandb

        self.logger.log({f"Bar Plots {epoch = }": wandb.Image(fig)})

        # Close the figure to free up memory
        plt.close(fig)

    def save_checkpoint(self, epoch, filename="model_checkpoint.pth"):
        """Saves model checkpoint."""
        # Create output folder
        result_folder = p.cwd()/'docs_and_results'
        current_time = datetime.now().strftime("%m-%d__%H-%M-%S")
        # current_time = '09-09__03-52-38'
        output_folder = utils_mnist_ssd.folder(result_folder/current_time)
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optim.state_dict(),
            "config": self.config
        }
        save_path = os.path.join(output_folder, filename)
        torch.save(checkpoint, save_path)
        time.sleep(3)
        print(f"Checkpoint saved at {save_path}")


    def fit(self):
        scheduler = self.get_scheduler()

        # Lists to store loss values per epoch
        total_losses = []
        ce_losses = []
        loc_losses = []
        ce_pos_losses = []
        ce_neg_losses = []

        for epoch in tqdm(range(self.start_epoch, self.config.epochs), desc=f"Training {self.config.model_name}", unit="iteration"):
            loss, ce_loss, loc_loss, ce_pos_loss, ce_neg_loss = self.train_step(epoch)

            # Store epoch-wise losses
            total_losses.append(loss.avg)
            ce_losses.append(ce_loss.avg)
            loc_losses.append(loc_loss.avg)
            ce_pos_losses.append(ce_pos_loss.avg)
            ce_neg_losses.append(ce_neg_loss.avg)

            # Run validation (if applicable)
            self.validation_step(epoch)
            # Save checkpoint
            if self.save_chpt_condition:
                self.save_checkpoint(epoch)
            # update previous performance
            self.prev_map = self.map
            self.prev_edit_distance = self.edit_distance

            if scheduler is not None:
                scheduler.step()
                print(f"{scheduler.get_last_lr() = }")

        return self.map, self.edit_distance
        # Plot the loss curves after training
        # self.plot_loss_curves(ce_losses, loc_losses, ce_pos_losses, ce_neg_losses)

    def plot_loss_curves(self, ce_losses, loc_losses, ce_pos_losses, ce_neg_losses, plot_local_curve=False):
        epochs = np.arange(1, len(ce_losses) + 1)

        # Convert lists to NumPy arrays
        ce_losses = np.array(ce_losses)
        loc_losses = np.array(loc_losses)
        ce_pos_losses = np.array(ce_pos_losses)
        ce_neg_losses = np.array(ce_neg_losses)
        if plot_local_curve:
            plt.figure(figsize=(10, 6))

            # Plot CE Loss
            plt.plot(epochs, ce_losses, label='Cross Entropy Loss', marker='o', linestyle='-')

            # Plot Localization Loss
            plt.plot(epochs, loc_losses, label='Localization Loss', marker='s', linestyle='-')

            # Plot CE Positive Loss
            plt.plot(epochs, ce_pos_losses, label='CE Positive Loss', marker='^', linestyle='--')

            # Plot CE Negative Loss
            plt.plot(epochs, ce_neg_losses, label='CE Negative Loss', marker='v', linestyle='--')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss Breakdown: CE Loss, Loc Loss, CE Pos Loss, CE Neg Loss')
            plt.legend()
            plt.grid()
            plt.show()

    def get_scheduler(self):
        configs = self.config
        if configs.scheduler_name == "MultiStepLR":
            # milestone
            ms = configs.multistep_milestones
            gamma = configs.multistep_gamma
            scheduler = MultiStepLR(self.optim, ms, gamma=gamma, verbose=True)
        elif configs.scheduler_name == "LinearLR":
            sf = self.config.linearLR_start_factor
            total_iter = self.config.linearLR_total_iter
            scheduler = LinearLR(self.optim, start_factor=sf, total_iters=total_iter)
        else:
            scheduler = None
        # console print
        if scheduler is None:
            print("No learning rate scheduler!")
        else:
            print(f"{scheduler} applied!")

        return scheduler

    def plot_bb(self, img_np, gt_boxes, matched_boxes, neg_boxes, predicted_boxes, epoch, i):
        # Image with bounding boxes
        fig, ax = plt.subplots(1, figsize=(8, 4))
        if self.config.color:
            ax.imshow(img_np)
        else:
            ax.imshow(img_np, cmap="gray")
        img_height, img_width, _ = img_np.shape

        # Get ground truth boxes and scale to image size
        gt_boxes[:, [0, 2]] *= img_width
        gt_boxes[:, [1, 3]] *= img_height

        # Get ground truth boxes and scale to image size
        # matched_boxes[:, [0, 2]] *= img_width
        # matched_boxes[:, [1, 3]] *= img_height

        # Get ground truth boxes and scale to image size
        if neg_boxes is not None:
            neg_boxes[:, [0, 2]] *= img_width
            neg_boxes[:, [1, 3]] *= img_height
        
        if predicted_boxes is not None:
            predicted_boxes = torch.Tensor(predicted_boxes)
            predicted_boxes[:, [0, 2]] *= img_width
            predicted_boxes[:, [1, 3]] *= img_height
            predicted_boxes = predicted_boxes.tolist()

        # Plot ground truth boxes
        # https://stackoverflow.com/questions/37435369/how-to-draw-a-rectangle-on-image
        for box in gt_boxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='pink', facecolor='none')
            ax.add_patch(rect)

        # Plot matched boxes
        # for box in matched_boxes:
        #     x_min, y_min, x_max, y_max = box
        #     rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='blue', facecolor='none')
        #     ax.add_patch(rect)
        if neg_boxes is not None:
            # Plot neg boxes
            for box in neg_boxes:
                x_min, y_min, x_max, y_max = box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

        if predicted_boxes is not None:
            # Plot neg boxes
            for box in predicted_boxes:
                x_min, y_min, x_max, y_max, lb = box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)
        ax.legend()
        ax.set_title(f"Image in epoch: {epoch}, step {i}")
        wandb.log({f"bbox_visual-{epoch =}": wandb.Image(fig)})
        plt.close(fig)
        del ax


def trainer(configs: ConfigParser, train_loader, val_loader, test_loader, logger, model_name):
    
    if model_name == "ssd_mnist":
        base_conv = BaseConv(configs.base_conv_conv_layers,
                    configs.base_conv_input_size, chosen_fm=[-2, -1],
                    norm=nn.BatchNorm2d, act_fn=nn.ReLU(), spectral=False)
        img, _, _ = next(iter(train_loader))
        base_size = pretty_print_module_list(base_conv.module_list, img)

        aux_conv = AuxConv(configs.aux_conv_conv_layers,
                        configs.aux_conv_input_size, norm=nn.BatchNorm2d, act_fn=nn.ReLU(), spectral=False)
        aux_size = pretty_print_module_list(aux_conv.module_list, torch.zeros(base_size[-1]))

        setattr(configs, 'fm_channels', [base_size[i][1] for i in base_conv.fm_id] + [aux_size[i][1] for i in aux_conv.fm_id])
        setattr(configs, 'fm_size', [base_size[i][-2:] for i in base_conv.fm_id] + [aux_size[i][-2:] for i in aux_conv.fm_id])
        setattr(configs, 'n_fm', len(configs.fm_channels))
        setattr(configs, 'fm_prior_aspect_ratio', configs.fm_prior_aspect_ratio[:configs.n_fm])
        setattr(configs, 'fm_prior_scale', np.linspace(0.1, 0.9, configs.n_fm))  # [0.2, 0.375, 0.55, 0.725, 0.9] # [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        assert len(configs.fm_prior_scale) == len(configs.fm_prior_aspect_ratio)
        setattr(configs, 'n_prior_per_pixel', [len(i)+1 for i in configs.fm_prior_aspect_ratio])  # in fm1, each pixel has 4 priors
        setattr(configs, 'multistep_milestones', list(range(10, configs.epochs, 5)))

        utils_mnist_ssd.img_size = [base_size[0][-2], base_size[0][-1]]
        new_h, new_w = configs.img_height//configs.downscale_factor, configs.img_width//configs.downscale_factor

        assert utils_mnist_ssd.img_size == [new_h, new_w], "mismatch!"

        model = SSD(configs, base_conv, aux_conv).to(configs.device)
        if configs.resume_from_checkpoint_path is not None:
            model.load_from_checkpoint()
        total_params = 0
        base_params = 0
        aux_params = 0
        pred_head_params = 0
        for param_name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{param_name}, {param.shape}")
                    if 'base_conv' in param_name:
                        base_params += param.numel()
                    elif 'aux_conv' in param_name:
                        aux_params += param.numel()
                    elif 'pred_conv' in param_name:
                        pred_head_params += param.numel()
                    
                    total_params += param.numel()
        print(f"{base_params = }, {aux_params = }, {pred_head_params = }")
        print(f"{(base_params + aux_params + pred_head_params) = }")
        print(f'{total_params = }')
        # a dummy forward pass to assert correctness
        locs, cls_, fms = model(img.to(configs.device))
        assert locs.shape[1] == cls_.shape[1], "mismatch!"
        # create optimizer, scheduler, criterion, then load checkpoint if specified
        bias = []
        not_bias = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'bias' in name:
                    bias.append(param)
                else:
                    not_bias.append(param)
        optimizer = torch.optim.Adam([{'params': bias, 'lr': configs.lr*2},
                                    {'params': not_bias}],
                                    lr=configs.lr, weight_decay=configs.weight_decay)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, configs.multistep_milestones, gamma=configs.multistep_gamma, verbose=False)
        
        loss_fn = MultiBoxLossSSD(priors_cxcy=model.priors_cxcy, configs=configs)

    elif model_name == "ssd_captcha":
        print(configs.base_conv_input_size)

        model = SSDCaptcha()
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        freeze_param_names = [
            "backbone.conv1_1.weight", "backbone.conv1_1.bias",
            "backbone.conv1_2.weight", "backbone.conv1_2.bias",
            "backbone.conv2_1.weight", "backbone.conv2_1.bias",
            "backbone.conv2_2.weight", "backbone.conv2_2.bias",
            "backbone.conv3_1.weight", "backbone.conv3_1.bias",
            "backbone.conv3_2.weight", "backbone.conv3_2.bias",
            "backbone.conv3_3.weight", "backbone.conv3_3.bias",
            "backbone.conv4_1.weight", "backbone.conv4_1.bias",
            "backbone.conv4_2.weight", "backbone.conv4_2.bias",
            "backbone.conv4_3.weight", "backbone.conv4_3.bias",
            "backbone.conv5_1.weight", "backbone.conv5_1.bias",
            "backbone.conv5_2.weight", "backbone.conv5_2.bias",
            "backbone.conv5_3.weight", "backbone.conv5_3.bias",
            "backbone.conv6.weight", "backbone.conv6.bias",
            "backbone.conv7.weight", "backbone.conv7.bias"
        ]
        freeze_backbone = False
        if freeze_backbone:
            freezed_params_groups = 0
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)
                if param_name in freeze_param_names:
                    freezed_params_groups += 1
                    print(f"freezing: {param_name}")
                    param.requires_grad = False
            print(f"freezed total of : {freezed_params_groups} groups")
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * configs.lr}, {'params': not_biases}],
                                    lr=configs.lr, momentum=configs.momentum, weight_decay=configs.weight_decay)

        # a dummy forward method to calculate the default boxes
        test_input = torch.randn(1, 3, 160, 640)  # Maintain rectangular aspect ratio
        pred_locs, pred_cls, fm_info = model(test_input)
        feature_map_shapes = fm_info.values()  # Example feature map sizes for rectangular input
        default_boxes = model.generate_default_boxes()
        loss_fn = MultiBoxLoss(default_boxes=default_boxes, config=configs)

    else:
        raise Exception(f"Invalid {model_name}!")

    # Train
    trainer = CaptchaTrainer(model, train_loader, val_loader, test_loader, loss_fn, optimizer, configs, logger)
    if configs.log_expt and configs.log_gradients:
        logger.watch(model, loss_fn, log_graph=True, log='all', log_freq=100)

    # train
    map_score = trainer.fit()
    return map_score
