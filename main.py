import sys
from pathlib import Path
from src_code.task_utils.config_parser import ConfigParser

def main(config_path: str | Path | None = None) -> None:
    # all the parameters can be obtained from this configs object
    configs = ConfigParser(config_path).get_parser()
    print(f"{configs.train_path = }")

    print("### Creating Dataloaders ###")

    print("### Training Model ###")
    
    print("### Evaluating Model ###")

if __name__ == "__main__":
    main()