import os
from pathlib import Path
import sys
import yaml


class ConfigParser:
    def __init__(self, config_path: str):
        self.config_dict = self.get_config(config_path)
        
    def get_parser(self):
        # ADD NEW CONFIG PARAMETERS HERE
        data_configs = self.config_dict.get("data_configs")
        if data_configs is None:
            raise Exception(f"data_configs is not available!")
        self.train_path = data_configs.get("train_path")
        return self
    
    def __verify__argparse(self, config_path):

        if config_path is None:
            args_count = len(sys.argv)
            if (args_count) > 2:
                print(f"One argument expected, got {args_count - 1}")
                raise SystemExit(2)
            elif args_count < 1:
                print("You must specify the config file")
                raise SystemExit(2)
            config_path = Path(sys.argv[1])
            return config_path
        print(f"{config_path } is being used!")
        
    def get_config(self, config_path: str):
        config_path = self.__verify__argparse(config_path)
        # reading from config file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config