from pathlib import Path
from src_code.task_utils.config_parser import ConfigParser
import wandb
from src_code.data_utils.dataset_utils import get_dataloader
from src_code.data_utils.dataset_utils import CaptchaDataset
from src_code.model_utils.train_utils import trainer


def main(config_path: str | Path | None = None) -> None:
    # all the parameters can be obtained from this configs object
    configs: ConfigParser = ConfigParser(config_path).get_parser()
    logger = None
    if configs.debug:
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
        config=configs
    )

    val_dataset = CaptchaDataset(
        configs.val_preprocessed_dir,
        configs.val_labels_dir,
        augment=False,
        config=configs
    )

    test_dataset = CaptchaDataset(
        configs.test_preprocessed_dir,
        labels_dir=None,
        augment=False,
        config=configs
    )

    # Create data loaders
    train_loader = get_dataloader(train_dataset, configs)
    val_loader = get_dataloader(val_dataset, configs)
    test_loader = get_dataloader(test_dataset, configs)
    img, bboxes, labels = next(iter(train_loader))
    # Print batch info
    print(f"Train Dataloader has {len(train_loader.dataset)} images")
    print(f"Validation Dataloader has {len(val_loader.dataset)} images")
    print(f"Test Dataloader has {len(test_loader.dataset)} images")

    print("### Training Model ###")
    trainer(configs,  train_loader, val_loader=val_loader, test_loader=test_loader, logger=logger)
    
    print("### Evaluating Model ###")
    # @todo dhimitri add your evaluation files here
    
    if configs.debug:
        # close wandb
        wandb.finish()

if __name__ == "__main__":
    main()
