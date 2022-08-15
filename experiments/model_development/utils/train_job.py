"""
train_job.py: contains the implementation of a model training job and its interface with the config file.
"""

# general imports
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import re, yaml, os
import pickle5 as pickle
import shutil
import random
from torch.utils.data import DataLoader
from datetime import datetime
from utils.framesdata import FramesDataset
from utils.translators import expts
from torch.autograd import Variable

# for training and validation
import os.path as osp
import mmcv
from mmcv import Config
from utils.models.Video_Swin_Transformer.mmaction.datasets import build_dataset
from utils.models.Video_Swin_Transformer.mmaction.models import build_model
from utils.models.Video_Swin_Transformer.mmaction.apis import train_model

# for testing
from utils.models.Video_Swin_Transformer.mmaction.apis import single_gpu_test
from utils.models.Video_Swin_Transformer.mmaction.datasets import build_dataloader
from mmcv.parallel import MMDataParallel
from mmcv.runner import set_random_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingConfig():
    """
    An intuitive way of translating config data from memory/disk into an object.
    """
    def __init__(self, data={}):
        """
        Initializes a Training Config from a dictionary of data.

        :param dict data: a dictionary describing the config. Nested dictionaries create nested config structures.
        """
        self.data = data
        for k,v in self.data.items():
            if type(v) == dict:
                # If nested dictionary, convert nested dicrionary into a config.
                setattr(self, k, TrainingConfig(v))
            else:
                # Otherwise, add it as an attirbute to the instance.
                setattr(self, k, v)

    def write_yaml(self, path):
        """
        Writes the content of the config into a yaml file.

        :param str path: path of yaml file to which the data is dumped.
        """
        with open(path, "w") as yaml_file:
            yaml.dump(self.data, yaml_file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Constructs a Training Config based on data from a yaml file.

        :param str yaml_path: path to a yaml file containing config.
        :return: a config instance initialized with the data from yaml
        :rtype: TrainingConfig
        """
        with open(yaml_path) as yaml_stream:
            parsed_yaml = yaml.safe_load(yaml_stream)
        return TrainingConfig(parsed_yaml)

class TrainingJob():
    """
    Trains and evaluates CNN+LSTM model based on a configuration file.
    """
    def __init__(self, config, stdout=True, using_ffcv=False, test=False):
        """
        Initialize the job and its parameters.

        :param TrainignConfig config: configuration for model training.
        :param function label_to_int: a function that translates labels from the dataset output into 0-indexed integers.
        :param bool using_ffcv: whether data is stored in FFCV format.
        :param bool stdout: whether to show training progress in stdout.
        """
        # Public training job attributes
        self.config = config
        self.using_ffcv = using_ffcv
        self.cnn_architecture = config.model.cnn_architecture
        self.stdout = stdout
        self.label_translator = expts[config.expt_name]

        # Output setup
        self._start_time = re.sub(r"[^\w\d-]", "_", str(datetime.now()))
        self._out_path = f"output/{self.config.job_name}_{self._start_time}"
        os.makedirs(self._out_path)
        self._log_path = os.path.join(self._out_path, "training.log")
        self._debug_path = os.path.join(self._out_path, "debugging.log")
        self._best_model_path = os.path.join(self._out_path, "model.pt")
        self.config.write_yaml(os.path.join(self._out_path, "config.yaml"))

        # Config setup
        self.cfg_file_path = self.config.model.config_file_path
        self.cfg = Config.fromfile(self.cfg_file_path)
        self.cfg.load_from = self.config.model.checkpoint_file_path
        self.cfg.seed = self.config.seed
        set_random_seed(self.cfg.seed, deterministic=False)
        self.cfg.model.cls_head.num_classes = self.config.model.num_classes
        self.cfg.setdefault('omnisource', False)

        # Dataset setup
        self.original_data_path = self.config.data_loader.original_data_path
        self.split_data_path = self.config.data_loader.split_data_path
        if not os.path.exists(self.split_data_path):
            os.makedirs(self.split_data_path, exist_ok=True)
            num_iterations = len(os.listdir(self.original_data_path))
            validation_iterations = random.sample(range(num_iterations), int((1 - self.config.data_loader.train_split) * num_iterations)) # TODO: account for case when test=True
            for iteration_dir in os.listdir(self.original_data_path):
                if iteration_dir == '.DS_Store':
                    continue

                iteration_data_path = os.path.join(self.original_data_path, iteration_dir, 'machine_readable', 'iteration_data.pickle')

                # get classification label
                with open(iteration_data_path, 'rb') as iteration_data_pickled:
                    iteration_data_unpickled = pickle.load(iteration_data_pickled)
                    label = self.label_translator(iteration_data_unpickled['label'])

                # rename video with iteration number and move .mp4 file to root of subdirectory
                old_video_name = os.path.join(self.original_data_path, iteration_dir, 'experiment_video.mp4')
                if int(iteration_dir) in validation_iterations:
                    os.makedirs(os.path.join(self.split_data_path, 'val'), exist_ok=True)
                    label_file_path = os.path.join(self.split_data_path, 'val_labels.txt')
                    new_video_name = os.path.join(self.split_data_path, 'val', iteration_dir + '_experiment_video.mp4')
                else: # TODO: account for case when test=True
                    os.makedirs(os.path.join(self.split_data_path, 'train'), exist_ok=True)
                    label_file_path = os.path.join(self.split_data_path, 'train_labels.txt')
                    new_video_name = os.path.join(self.split_data_path, 'train', iteration_dir + '_experiment_video.mp4')
                os.rename(old_video_name, new_video_name)

                # write video name and label as new line in label .txt file
                with open(label_file_path, 'a+') as label_file:
                    label_file.write(os.path.basename(new_video_name) + ' ' + str(label))
                    label_file.write('\n')

        # Dataset root directories and annotations setup
        self.cfg.dataset_type = 'VideoDataset'
        self.cfg.data_root = os.path.join(self.split_data_path, 'train/')
        self.cfg.data_root_val = os.path.join(self.split_data_path, 'val/')
        self.cfg.ann_file_train = os.path.join(self.split_data_path, 'train_labels.txt')
        self.cfg.ann_file_val = os.path.join(self.split_data_path, 'val_labels.txt')
        # Training dataset setup
        self.cfg.data.train.type = os.path.join('VideoDataset')
        self.cfg.data.train.ann_file = os.path.join(self.split_data_path, 'train_labels.txt')
        self.cfg.data.train.data_prefix = os.path.join(self.split_data_path, 'train/')

        # Validation dataset setup
        self.cfg.data.val.type = os.path.join('VideoDataset')
        self.cfg.data.val.ann_file = os.path.join(self.split_data_path, 'val_labels.txt')
        self.cfg.data.val.data_prefix = os.path.join(self.split_data_path, 'val/')

        # Testing data setup
        if test:
            self.cfg.ann_file_test = os.path.join(self.split_data_path, 'test_labels.txt')
            self.cfg.data.test.type = os.path.join('VideoDataset')
            self.cfg.data.test.ann_file = os.path.join(self.split_data_path, 'test_labels.txt')
            self.cfg.data.test.data_prefix = os.path.join(self.split_data_path, 'test/')

        # File and log save location setup
        self.cfg.work_dir = self._out_path

        # The original learning rate (LR) is set for 8-GPU training.
        # We divide it by 8 since we only use one GPU.
        self.cfg.data.videos_per_gpu = max(self.cfg.data.videos_per_gpu // 16, 1)
        self.cfg.optimizer.lr = self.cfg.optimizer.lr / 8 / 16
        self.cfg.total_epochs = 30
        self.cfg.gpu_ids = range(1)


    def train(self):
        """
        Runs the training job by training the model on the training data.

        :param bool evaluate: whether to evaluate the model at each epoch and save the best model.
        :return: training and testing accuracies and losses.
        :rtype: dict["train":tuple(float, float), "test":tuple(float, float)]
        """

        # Build the dataset
        datasets = [build_dataset(self.cfg.data.train)]

        # Build the recognizer
        model = build_model(self.cfg.model, train_cfg=self.cfg.get('train_cfg'), test_cfg=self.cfg.get('test_cfg'))

        # Create work_dir
        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))
        train_model(model, datasets, self.cfg, distributed=False, validate=True)

    def test(self):
        dataset = build_dataset(self.cfg.data.test, dict(test_mode=True))
        data_loader = build_dataloader(
                dataset,
                videos_per_gpu=1,
                workers_per_gpu=self.cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False)
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)

        eval_config = self.cfg.evaluation
        eval_config.pop('interval')
        eval_res = dataset.evaluate(outputs, **eval_config)
        for name, val in eval_res.items():
            print(f'{name}: {val:.04f}')

if __name__ == '__main__':
    config = TrainingConfig.from_yaml("config/ModelArchitecture.yaml")
    job = TrainingJob(config=config)
