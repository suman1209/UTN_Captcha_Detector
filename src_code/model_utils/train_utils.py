import time
from abc import ABC, abstractmethod
import torch.backends.cudnn as cudnn
from torch import nn, optim
import torch.utils.data
from torch.utils.data import DataLoader
from loss import MultiBoxLoss
from ssd import SSD300
from data_utils.dataset_utils import CaptchaDataset
from data_utils.preprocessing import *



# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True




class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.crit = loss_fn
        self.optimizer = optimizer
        self.device = device

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accs: list[float] = []
        self.configs: AppConfigs = AppConfigs.get_config()

    def fit(self, epochs: int, verbose=False):
        for epoch in range(epochs):
            self.train_epoch(epoch, verbose)
            self.val_epoch(epoch, verbose)
            writer.flush()
            checkpoint_epoch = self.configs.checkpoint_count

            if epoch % checkpoint_epoch == 0 and epoch != 0:
                save_model(model=self.model, model_path=f"{self.configs.output_dir}/model_checkpoint_{epoch}.pt")
                losses_history_path = f'{self.configs.output_dir}/losses_hist_checkpoint_{epoch}.pkl'
                losses_history_dict = {'train_loss': self.train_losses, 'val_loss': self.val_losses}
                save_as_pickle(losses_history_dict, losses_history_path)

        print(
            f"Finished Training!:\n"
            f"Train Loss: {self.train_losses[-1]:.2f},\n"
            f"Val Loss: {self.val_losses[-1]:.2f},\n"
        )
        writer.close()

    @abstractmethod
    def train_epoch(self, epoch: int, verbose: bool):
        ...

    @abstractmethod
    def val_epoch(self, epoch: int, verbose: bool):


class SDCTrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: nn.Module,
                 optimizer: optim,
                 device: torch.device):
        super().__init__(model, train_loader, val_loader, loss_fn, optimizer, device)

    def train_epoch(self, epoch: int, verbose: bool = False):
        self.model.train()
        total_loss = 0.0
        for datapoints in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            if not self.configs.use_all_three_cameras:
                # print(f"Using only the center Image")
                """if only a single camera image is used"""
                inputs, targets = datapoints[0]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.crit(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            else:
                # print(f"Using all the three left, center and right images")
                """if all the three camera images are used for the training"""
                for datapoint in datapoints:
                    inputs, targets = datapoint
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.crit(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        # update tensorboard statistics
        writer.add_scalar("Loss/train", avg_loss, epoch)
        # log the input images to the model
        img_grid = torchvision.utils.make_grid(inputs)
        writer.add_image(f'Input Images to the model @epoch{epoch}', img_grid)
        if verbose:
            print(f"Epoch: {epoch}, Train loss: {avg_loss}")

    def val_epoch(self, epoch: int, verbose: bool = False):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for datapoint in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                inputs, targets = datapoint[0]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.crit(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        # update tensorboard statistics
        writer.add_scalar("Loss/val", avg_loss, epoch)
        if verbose:
            print(f"Epoch: {epoch}, Val loss: {avg_loss}")


if __name__ == "__main__":
    pass




def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()