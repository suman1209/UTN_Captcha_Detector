import time
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
                 logger = None
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

            # Backward pass
            loss.backward()

            if self.logger is not None:
                self.logger.log({"train_loss": loss})
                # @todo add more things whenever needed
                # self.logger.log({"ce_loss": debug_info['ce_loss']})
                # self.logger.log({"loc_loss": debug_info['loc_loss']})
                # self.logger.log({"ce_pos_loss": debug_info['ce_pos_loss']})
                # self.logger.log({"ce_neg_loss": debug_info['ce_hard_neg_loss']})
                # free_mem, avail_mem = torch.cuda.mem_get_info(device=None)
                # free_mem = free_mem / 1e9
                # avail_mem = avail_mem / 1e9
                # self.logger.log({"gpu_free_mem": free_mem})
    
            self.optim.step()
            losses.update(loss.item(), images.size(0))
            # ce_losses.update(debug_info['ce_loss'], images.size(0))
            # loc_losses.update(debug_info['loc_loss'], images.size(0))
            # ce_pos_losses.update(debug_info['ce_pos_loss'], images.size(0))
            # ce_neg_losses.update(debug_info['ce_hard_neg_loss'], images.size(0))
            if i % self.config.print_freq == 0:                
                print(f"Epoch: {epoch} avg Loss per epoch: {losses.avg:.4f}")
                # print(f"{debug_info = }")
            
            del loc_pred, cls_pred, images, boxes, labels, debug_info
            torch.cuda.empty_cache()
        return losses

    def validation_step(self):
        self.model.eval()
        self.model.to(self.config.device)
        with torch.no_grad():
            for i, (images, boxes, labels) in enumerate(self.test_loader):
                images = images.to(self.config.device) 
                locs_pred, cls_pred, fm_info = self.model(images)
                # @todo need to add the evaluation methods here.

    def fit(self):
        for epoch in range(self.start_epoch, self.config.get_parser().epochs):
            self.train_step(epoch)
            self.validation_step()
        
def trainer(configs: ConfigParser, train_loader, val_loader, test_loader, logger):
    model = SSDCaptcha()
    # Initialize model or load checkpoint
    if not hasattr(configs, "checkpoint") or configs.checkpoint is None:
        start_epoch = 0
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
    else:
        raise Exception("No support for checkpoint as of now!")
    # a dummy forward method to calculate the default boxes
    
    test_input = torch.randn(1, 1, 40, 160)  # Maintain rectangular aspect ratio
    pred_locs, pred_cls, fm_info = model(test_input)
    feature_map_shapes = fm_info.values()  # Example feature map sizes for rectangular input
    default_boxes = model.generate_default_boxes(feature_map_shapes)
    loss_fn = MultiBoxLoss(default_boxes=default_boxes, config=configs)

    trainer = CaptchaTrainer(model, train_loader, val_loader, test_loader, loss_fn, optimizer, configs, logger)
    
    # train
    trainer.fit()