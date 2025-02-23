from pathlib import Path
from src_code.task_utils.config_parser import ConfigParser
import wandb
from src_code.data_utils.dataset_utils import get_dataloader
from src_code.data_utils.dataset_utils import CaptchaDataset
from src_code.model_utils.train_utils import trainer
from src_code.data_utils.preprocessing import get_img_transform, get_rectangle_img_transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from pathlib import Path as p
from datetime import datetime
import yaml


def main(config_path: str | Path | None = None) -> None:
    # all the parameters can be obtained from this configs object
    configs: ConfigParser = ConfigParser(config_path).get_parser()

    with open('./configs/default_ssd_configs.yaml', 'r') as file:
        default_model_configs = yaml.safe_load(file)
    assert isinstance(default_model_configs, dict)
    configs.update(default_model_configs)
    new_h = configs.img_height // configs.downscale_factor
    new_w = configs.img_width // configs.downscale_factor
    setattr(configs, "base_conv_input_size", [new_h, new_w])
    logger = None

    if configs.log_expt:
        # wandb initialisation
        wandb.init(
        # set the wandb project where this run will be logged
        project="utn-captcha-detector",

        # track hyperparameters and run metadata
        config=configs)
        logger = wandb

    print(f"{configs.batch_size = }")
    print("### Creating Dataloaders ###")

    # Create datasets
    train_dataset = CaptchaDataset(
        configs.train_preprocessed_dir,
        configs.train_labels_dir,
        augment=True,
        config=configs,
        img_transform=get_rectangle_img_transform(configs)
    )
    
    val_dataset = CaptchaDataset(
        configs.val_preprocessed_dir,
        configs.val_labels_dir,
        augment=False,
        config=configs,
        img_transform=get_rectangle_img_transform(configs)
    )

    test_dataset = CaptchaDataset(
        configs.test_preprocessed_dir,
        labels_dir=None,
        augment=False,
        config=configs,
        img_transform=get_rectangle_img_transform(configs)
    )

    # Create data loaders
    train_loader = get_dataloader(train_dataset, configs)
    val_loader = get_dataloader(val_dataset, configs)
    test_loader = get_dataloader(test_dataset, configs)
    img, bboxes, labels = next(iter(train_loader))
    # Print batch info
    train_img_count = len(train_loader.dataset)
    print(f"Train Dataloader has {train_img_count} images")
    print(f"Validation Dataloader has {len(val_loader.dataset)} images")
    print(f"Test Dataloader has {len(test_loader.dataset)} images")
    assert train_img_count > configs.batch_size, f"Only {train_img_count} train_imgs, {configs.batch_size=}"
    print("### Training Model ###")
    trainer(configs,  train_loader, val_loader=val_loader, test_loader=test_loader,
            logger=logger, model_name=configs.model_name)
    if configs.log_expt:
        # close wandb
        wandb.finish()
    
    
if __name__ == "__main__":
    main()
