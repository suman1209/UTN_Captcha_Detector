from pathlib import Path
from src_code.task_utils.config_parser import ConfigParser
import wandb
from src_code.data_utils.dataset_utils import get_dataloader
from src_code.data_utils.dataset_utils import CaptchaDataset



def main(config_path: str | Path | None = None) -> None:
    # all the parameters can be obtained from this configs object
    configs: ConfigParser = ConfigParser(config_path).get_parser()
    if configs.debug:
        # wandb initialisation
        wandb.init(
        # set the wandb project where this run will be logged
        project="utn-captcha-detector",

        # track hyperparameters and run metadata
        config=configs)

    print(f"{configs.train_path = }")

    print("### Creating Dataloaders ###")

    train_dataset = CaptchaDataset(configs)
    train_dataloader = get_dataloader(train_dataset, configs)

    # Print batch info
    print(f"Dataloader has {len(train_dataloader.dataset)} images")


    print("### Training Model ###")
    # @todo, the following will go into the training script as a parameter
    if configs.debug:
        loss = -1
        wandb.log({"loss": loss})
    print("### Evaluating Model ###")
    
    if configs.debug:
        # close wandb
        wandb.finish()

if __name__ == "__main__":
    main()
