from pathlib import Path
from src_code.task_utils.config_parser import ConfigParser
import wandb
from src_code.data_utils.dataset_utils import get_dataloader
from src_code.data_utils.dataset_utils import CaptchaDataset
from src_code.model_utils.train_utils import trainer
from src_code.data_utils.preprocessing import get_rectangle_img_transform, preprocess_all
import yaml
import sys


def main2(config_path: str | Path | None = None) -> None:
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

    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: python main.py <config_path>")
            sys.exit(1)
        config_path = sys.argv[1]  # Read config path from command line

    config_path = str(config_path)  # Convert Path object to string
    print(f"Using config: {config_path}")

    print("### Run Preprocessing ###")   
    preprocess_all(config_path)

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

    if configs.task == 'train':
        map_score = trainer(configs,  train_loader, val_loader=val_loader, test_loader=test_loader, 
                            logger=logger, model_name=configs.model_name)

    elif configs.task == 'sweep':
        # 1: Define objective/training function
        def objective(config):
            # update the configs files
            configs.batch_size = config.batch_size
            configs.lr = config.lr
            configs.alpha = config.alpha
            configs.hard_neg_pos = config.hard_neg_pos
            configs.flip_prob = config.flip_prob
            configs.zoom_prob = config.zoom_prob
            configs.rotation_prob = config.rotation_prob
            configs.log_expt = False
            configs.epochs = 10
            map_score, edit_dist = trainer(configs,  train_loader, val_loader=val_loader, test_loader=test_loader,
                            logger=logger, model_name=configs.model_name)
            return edit_dist

        def main():
            wandb.init(project="Captcha-sweep")
            edit_dist = objective(wandb.config)
            wandb.log({"edit_dist": edit_dist})

        # 2: Define the search space
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "minimize", "name": "edit_dist"},
            "parameters": {
                "batch_size": {"values": [16, 32, 48, 64]},
                "lr": {"values": [1e-2, 1e-3, 1e-4]},
                "alpha": {"values": [0.1, 0.25, 0.5, 0.75, 1.0]},
                "hard_neg_pos": {"values": [0.5, 1, 2, 3, 4, 5]},
                "rotation_prob": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
                "flip_prob": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
                "zoom_prob":{"values": [0.1, 0.2, 0.3, 0.4, 0.5]}
            },
        }

        # 3: Start the sweep
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="Captcha-sweep")

        wandb.agent(sweep_id, function=main, count=10)

        if configs.log_expt:
            # close wandb
            wandb.finish()
    # @dhimitri, can you update this
    elif configs.task == 'evaluate':
        # generate a report of the following metrics in the val and test datasets
        # 1. Evaluate mAP of best model
        # 2. Evaluate Edit Distance (maybe average?)
        raise Exception(f'This is yet to be implemented!')

    else:
        raise Exception(f'Undefined task! {configs.task}')

if __name__ == "__main__":
    main2()
