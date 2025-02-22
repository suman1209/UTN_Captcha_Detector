import time
import random
from abc import ABC, abstractmethod
import torch.backends.cudnn as cudnn
from torch import nn, optim
import torch.utils.data
from torch.utils.data import DataLoader
from src_code.model_utils.loss import MultiBoxLoss
from src_code.model_utils.ssd import SSDCaptcha
from src_code.data_utils.dataset_utils import CaptchaDataset
from src_code.data_utils.preprocessing import *
from src_code.task_utils.config_parser import ConfigParser
from src_code.data_utils.dataset_utils import category_id_labels
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
import os
os.environ['WANDB_CACHE_DIR'] = "./"
os.environ['WANDB_DATA_DIR'] = "./"
import numpy as np


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
                 logger = None, checkpoint_path = "checkpoints"
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

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Load checkpoint if available
        if hasattr(config, "checkpoint") and config.checkpoint is not None:
            self.load_checkpoint(config.checkpoint)

    def train_step(self, epoch):
        self.model.to(self.config.device)
        losses = Metrics()
        ce_losses = Metrics()
        loc_losses = Metrics()
        ce_pos_losses = Metrics()
        ce_neg_losses = Metrics()
        assert len(self.train_loader) > 0, f"{len(self.train_loader) = }"

        for i, (images, boxes, labels) in enumerate(self.train_loader):
            images = images.to(self.config.device)  # (batch_size (N), 3, 160, 640)
            images.requires_grad=True
            self.optim.zero_grad()
            # Foward pass
            loc_pred, cls_pred, fm_info = self.model(images)
            # loss
            loss, debug_info = self.loss_fn(loc_pred, cls_pred, boxes, labels)

            
            # Extract and detach loss components
            ce_loss = debug_info.get('ce_loss', torch.tensor(0.0, device=self.config.device)).detach().cpu().item()
            loc_loss = debug_info.get('loc_loss', torch.tensor(0.0, device=self.config.device)).detach().cpu().item()
            ce_pos_loss = debug_info.get('ce_pos_loss', torch.tensor(0.0, device=self.config.device)).detach().cpu().item()
            ce_neg_loss = debug_info.get('ce_hard_neg_loss', torch.tensor(0.0, device=self.config.device)).detach().cpu().item()
            loss_value = loss.detach().cpu().item()

            # Backward pass
            loss.backward()

            self.optim.step()

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
                # @todo add more things whenever needed
                # self.logger.log({"ce_loss": debug_info['ce_loss']})
                # self.logger.log({"loc_loss": debug_info['loc_loss']})
                # self.logger.log({"ce_pos_loss": debug_info['ce_pos_loss']})
                # self.logger.log({"ce_neg_loss": debug_info['ce_hard_neg_loss']})
                # free_mem, avail_mem = torch.cuda.mem_get_info(device=None)
                # free_mem = free_mem / 1e9
                # avail_mem = avail_mem / 1e9
                # self.logger.log({"gpu_free_mem": free_mem})
    
            # losses.update(loss.item(), images.size(0))
            # ce_losses.update(debug_info['ce_loss'], images.size(0))
            # loc_losses.update(debug_info['loc_loss'], images.size(0))
            # ce_pos_losses.update(debug_info['ce_pos_loss'], images.size(0))
            # ce_neg_losses.update(debug_info['ce_hard_neg_loss'], images.size(0))
            if i % self.config.print_freq == 0:
                print(f"Epoch: {epoch} | Loss: {losses.avg:.4f} | CE Loss: {ce_losses.avg:.4f} | "
                    f"Loc Loss: {loc_losses.avg:.4f} | CE Pos Loss: {ce_pos_losses.avg:.4f} | "
                    f"CE Neg Loss: {ce_neg_losses.avg:.4f}")
                
            del loc_pred, cls_pred, images, boxes, labels, debug_info
            torch.cuda.empty_cache()
        return losses, ce_losses, loc_losses, ce_pos_losses, ce_neg_losses

    def validation_step(self, epoch):
        self.model.eval()
        self.model.to(self.config.device)
        only_once = False
        with torch.no_grad():
            for i, (images, boxes, labels) in enumerate(self.val_loader):
                images = images.to(self.config.device) 
                loc_pred, cls_pred, fm_info = self.model(images)
                loss, debug_info = self.loss_fn(loc_pred, cls_pred, boxes, labels)
                str_labels = ["".join([category_id_labels[i.item()] for i in label]) for label in labels]
                if not only_once:
                    # rand image and draw the ground truth bbox and the matched default box
                    random_image = 0#random.randint(0, images.shape[0] - 1)
                    img_np = images[random_image].cpu().detach().numpy().transpose(1, 2, 0)
                    label = str_labels[random_image]
                    gt_boxes = boxes[random_image].cpu().numpy()

                    # matched_boxes = debug_info['matched_gt_boxes'][random_image].cpu().numpy()
                    # to draw all the positive boxes that have been matched
                    matched_boxes = debug_info["pos_default_boxes"][random_image].cpu().numpy()
                    neg_boxes = debug_info["hardneg_default_boxes"][random_image].cpu().numpy()
                    pb = len(matched_boxes)
                    self.plot_bb(img_np, gt_boxes, matched_boxes, neg_boxes, f"epoch={epoch} label = {label} num_pos_boxes={pb}", i)
                    logits = debug_info["soft_maxed_pred"][random_image]
                    GT_int = labels[random_image].tolist()
                    GT_str = str_labels[random_image]
                    my_table = wandb.Table(columns=["GT"] + list(category_id_labels.values()) + ["bg"], data=[[f"{GT_str[i]}-{GT_int[i]}"] + logit.tolist() for i, logit in enumerate(logits)])
                    # self.logger.log({f"logits outputed: {epoch = }": my_table})
                    self.log_logits(my_table, epoch)
                break

    def log_logits(self, my_table, epoch):
        """this code was generated using deepseek with the following prompt: 
        
        my_table = wandb.Table(columns=["GT"] + list(category_id_labels.values()) + ["bg"], data=[[f"{GT_str[i]}-{GT_int[i]}"] + logit.tolist() for i, logit in enumerate(logits)])
        i want a bar plot for each row, and all the bar plots in a single image divided by grids
        
        """
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
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optim.state_dict(),
            "config": self.config
        }
        save_path = os.path.join(self.checkpoint_path, filename)
        torch.save(checkpoint, save_path)
        # print(f"Checkpoint saved at {save_path}")

    def load_checkpoint(self, filename="model_checkpoint.pth"):
        """Loads model checkpoint if available."""
        # Ensure the filename is a correct absolute path
        if not os.path.isabs(filename):  
            load_path = filename
        else:
            load_path = os.path.join("checkpoints", filename) 

        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optim.load_state_dict(checkpoint["optimizer_state"])
            self.start_epoch = checkpoint["epoch"] + 1
            print(f"Checkpoint loaded from {load_path} (starting from epoch {self.start_epoch})")
        else:
            print(f"No checkpoint found at {load_path}. Training from scratch.")
    
    def fit(self):
        scheduler = self.get_scheduler()

        # Lists to store loss values per epoch
        total_losses = []
        ce_losses = []
        loc_losses = []
        ce_pos_losses = []
        ce_neg_losses = []

        for epoch in range(self.start_epoch, self.config.epochs):
            loss, ce_loss, loc_loss, ce_pos_loss, ce_neg_loss = self.train_step(epoch)

            # Store epoch-wise losses
            total_losses.append(loss.avg)
            ce_losses.append(ce_loss.avg)
            loc_losses.append(loc_loss.avg)
            ce_pos_losses.append(ce_pos_loss.avg)
            ce_neg_losses.append(ce_neg_loss.avg)

            # Save checkpoint at every epoch
            # self.save_checkpoint(epoch)
            
            # Run validation (if applicable)
            self.validation_step(epoch)

            if scheduler is not None:
                scheduler.step(epoch)
                # print(f"{scheduler.get_last_lr() = }")

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
            scheduler = MultiStepLR(self.optim, ms, gamma=gamma , verbose=True)
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
    
    def plot_bb(self, img_np, gt_boxes, matched_boxes, neg_boxes, epoch, i):
        # Image with bounding boxes
        fig, ax = plt.subplots(1, figsize=(8, 4))
        ax.imshow(img_np, cmap="gray")

        img_height, img_width, _ = img_np.shape

        # Get ground truth boxes and scale to image size
        gt_boxes[:, [0, 2]] *= img_width
        gt_boxes[:, [1, 3]] *= img_height
        
        
        # Get ground truth boxes and scale to image size
        matched_boxes[:, [0, 2]] *= img_width
        matched_boxes[:, [1, 3]] *= img_height

        # Get ground truth boxes and scale to image size
        neg_boxes[:, [0, 2]] *= img_width
        neg_boxes[:, [1, 3]] *= img_height
        
        # Plot ground truth boxes
        # https://stackoverflow.com/questions/37435369/how-to-draw-a-rectangle-on-image
        for box in gt_boxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='pink', facecolor='none')
            ax.add_patch(rect)

        # Plot matched boxes
        for box in matched_boxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
        
        # Plot neg boxes
        for box in neg_boxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.legend()
        ax.set_title(f"Image in epoch: {epoch}, step {i}")
        wandb.log({f"bbox_visual-{epoch =}": wandb.Image(fig)})
        plt.close(fig)
    
def trainer(configs: ConfigParser, train_loader, val_loader, test_loader, logger):
    model = SSDCaptcha()
    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * configs.lr}, {'params': not_biases}],
                                lr=configs.lr, momentum=configs.momentum, weight_decay=configs.weight_decay)
        
    # # a dummy forward method to calculate the default boxes
    # test_input = torch.randn(1, 1, 40, 160)  # Maintain rectangular aspect ratio
    # pred_locs, pred_cls, fm_info = model(test_input)
    # feature_map_shapes = fm_info.values()  # Example feature map sizes for rectangular input
    default_boxes = model.generate_default_boxes()
    loss_fn = MultiBoxLoss(default_boxes=default_boxes, config=configs)

    trainer = CaptchaTrainer(model, train_loader, val_loader, test_loader, loss_fn, optimizer, configs, logger)
    
    # train
    trainer.fit()