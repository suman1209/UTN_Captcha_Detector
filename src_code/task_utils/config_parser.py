from pathlib import Path
import sys
import yaml


class ConfigParser:
    def __init__(self, config_path: str | dict):
        self.config_dict = self.get_config(config_path)

    def get_parser(self):
        # ADD NEW CONFIG PARAMETERS HERE
        # data configs
        self.task = self.config_dict.get("task")
        data_configs = self.config_dict.get("data_configs")
        if data_configs is None:
            raise Exception("data_configs is not available!")
        self.train_path = data_configs.get("train_path")
        prep_config = data_configs.get("preprocessing_related")
        self.downscale_factor = prep_config.get("downscale_factor")
        self.color = prep_config.get('color', True)

        # dataset configs
        dataset_config = data_configs.get("dataset_related")
        self.train_preprocessed_dir = dataset_config.get("train_preprocessed_dir")
        self.val_preprocessed_dir = dataset_config.get("val_preprocessed_dir")
        self.test_preprocessed_dir = dataset_config.get("test_preprocessed_dir")
        self.train_labels_dir = dataset_config.get("train_labels_dir")
        self.val_labels_dir = dataset_config.get("val_labels_dir")
        self.augment = dataset_config.get("augment")
        self.shuffle = dataset_config.get("shuffle")

        # augmentation configs
        aug_config = data_configs.get("augmentation_related")
        self.flip_prob = aug_config.get("flip_prob")
        self.zoom_prob = aug_config.get("zoom_prob")
        self.rotation_prob = aug_config.get("rotation_prob")
        self.line_prob = aug_config.get("line_prob")
        self.salt_pepper_prob = aug_config.get("salt_pepper_prob")

        # model configs
        model_configs = self.config_dict.get("model_configs")
        self.model_name = model_configs.get("name")
        self.log_gradients = model_configs.get("log_gradients")
        self.resume_from_checkpoint_path = model_configs.get("resume_from_checkpoint_path")
        if model_configs is None:
            raise KeyError("model_configs is missing from the config file!")
        self.model_configs = model_configs  
        self.checkpoint = model_configs.get("checkpoint", None)
        self.device = model_configs.get('device')
        self.print_freq = model_configs.get('print_freq')
        self.batch_size = model_configs.get("batch_size")
        self.epochs = model_configs.get("epochs")
        scheduler_configs = model_configs.get("scheduler")
        self.scheduler_name = scheduler_configs.get("name", None)
        self.multistep_milestones = scheduler_configs.get("milestones", None)
        self.multistep_gamma = scheduler_configs.get("gamma", None)
        self.linearLR_start_factor = scheduler_configs.get("start_factor", None)
        self.linearLR_total_iter = scheduler_configs.get("total_iter", None)
        loss_configs = model_configs.get("loss")
        self.pos_box_threshold = loss_configs.get("pos_box_threshold")
        self.neg_pos_hard_mining = loss_configs.get("hard_neg_pos")
        self.alpha = loss_configs.get("alpha")

        optim_configs = model_configs.get("optim")
        self.lr = optim_configs.get("lr")
        self.momentum = optim_configs.get("momentum")
        self.weight_decay = optim_configs.get("weight_decay")
        self.clip_grad = optim_configs.get("clip_grad")

        # task_configs
        task_configs = self.config_dict.get("task_configs")
        self.debug = task_configs.get("debug")
        self.log_expt = task_configs.get("log_expt")
        self.num_classes = task_configs.get("num_classes")
        self.img_height = task_configs.get("img_height")
        self.img_width = task_configs.get("img_width")
        self.nms_min_cls_score = task_configs.get("nms_min_cls_score")
        self.nms_iou_score = task_configs.get("nms_iou_score")
        self.nms_topk = task_configs.get("nms_topk")
        self.img_width = task_configs.get("img_width")
        return self

    def update(self, additional_config: dict):
        for key, val in additional_config.items():
            setattr(self, key, val)

    def __verify__argparse(self, config_path):

        if isinstance(config_path, str) or config_path is None:
            args_count = len(sys.argv)
            if (args_count) > 2:
                print(f"One argument expected, got {args_count - 1}")
                raise SystemExit(2)
            elif args_count <= 1:
                print("You must specify the config file")
                raise SystemExit(2)

            config_path = Path(sys.argv[1])
            return config_path
        elif isinstance(config_path, dict):
            return config_path
        print(f"{config_path } is being used!")

    def get_config(self, config: str | dict):
        config = self.__verify__argparse(config)
        print(f"{config = }")
        if isinstance(config, (str, Path)):
            # reading from yaml config file
            with open(config, 'r') as file:
                config_dict = yaml.safe_load(file)
        elif isinstance(config, dict):
            config_dict = config
        return config_dict
