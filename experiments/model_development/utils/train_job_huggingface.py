"""
train_job.py: contains the implementation of a model training job and its interface with the config file.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import re, yaml, os

from utils.framesdata import FramesDataset, collate_videos
from utils.model import CNNLSTM
from utils.translators import expts, label_keys

import matplotlib.pyplot as plt

from transformers import VideoMAEForPreTraining, AutoImageProcessor, TrainingArguments, Trainer

import pytorchvideo.data

from pytorchvideo.transforms import ApplyTransformToKey, Normalize, RandomShortSideScale, RemoveKey, ShortSideScale, UniformTemporalSubsample

from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip, Resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

class TrainingConfig:
    """
    An intuitive way of translating config data from memory/disk into an object.
    """

    def __init__(self, data={}):
        """
        Initializes a Training Config from a dictionary of data.

        :param dict data: a dictionary describing the config. Nested dictionaries create nested config structures.
        """
        self.data = data
        for k, v in self.data.items():
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


class TrainingJob:
    """
    Trains and evaluates CNN+LSTM model based on a configuration file.
    """

    def __init__(self, config, stdout=True, using_ffcv=False, ckpt_path=None):
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
        self.stdout = stdout
        self.label_translator = expts[config.expt_name]

        # Output set up
        self._start_time = re.sub(r"[^\w\d-]", "_", str(datetime.now()))
        
        out_path = f"output/{self.config.job_name}_{self._start_time}"
        os.makedirs(out_path)
        self._log_path = os.path.join(out_path, "training.log")
        self._debug_path = os.path.join(out_path, "debugging.log")
        self.config.write_yaml(os.path.join(out_path, "config.yaml"))
        
        ckpts_path = os.path.join(out_path, "ckpts")
        os.makedirs(ckpts_path)
        self._best_model_path = os.path.join(ckpts_path, "best.ckpt")
        self._epoch_model_path = os.path.join(ckpts_path, "ep%i.ckpt")
        
        self.writer = SummaryWriter(os.path.join(out_path, "tensorboard"))

        # Setting up data loaders, the model, and the optimizer & loss function
        self.model_ckpt = self.config.model.model_checkpoint
        self.model = VideoMAEForPreTraining.from_pretrained(
            self.model_ckpt
        ).to(device)
        self.args = TrainingArguments(
            self.config.job_name,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.train_params.lr,
            per_device_train_batch_size=self.config.data_loader.batch_size,
            per_device_eval_batch_size=self.config.data_loader.batch_size,
            logging_steps=1,
            metric_for_best_model="accuracy"
        )

        self.image_processor = AutoImageProcessor.from_pretrained(self.model_ckpt) # image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.train_dataset, self.val_dataset = self.get_datasets()
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.config.train_params.lr
        # )

        # Initializing log and log metadata
        self._log(f"Starting Log, {self.config.job_name} + {self.config.expt_name}")

        # Keep count of samples seen in training
        self.train_count = 0

    def train(self, evaluate=False):
        """
        Runs the training job by training the model on the training data.

        :param bool evaluate: whether to evaluate the model at each epoch and save the best model.
        :return: training and testing accuracies and losses.
        :rtype: dict["train":tuple(float, float), "test":tuple(float, float)]
        """

        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset
        )

        train_results = self.trainer.train()

    def get_datasets(self):
        mean = self.image_processor.image_mean
        std = self.image_processor.image_std
        if "shortest_edge" in self.image_processor.size:
            height = width = self.image_processor.size["shortest_edge"]
        else:
            height = self.image_processor.size["height"]
            width = self.image_processor.size["width"]
        resize_to = (height, width)

        num_frames_to_sample = self.model.config.num_frames
        sample_rate = 4
        fps = 30
        clip_duration = num_frames_to_sample * sample_rate / fps

        train_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            )
        ])

        val_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ])

        train_dataset = FramesDataset(
            self.config.data_loader.train_data_path,
            self.label_translator,
            fpv=None,
            skip_every=self.config.data_loader.skip_every,
            train=True,
            shuffle=True,
            source_type=self.config.data_loader.source_type,
            yaml_label_key=label_keys[self.config.expt_name],
        )

        val_dataset = FramesDataset(
            self.config.data_loader.val_data_path,
            self.label_translator,
            fpv=None,
            skip_every=self.config.data_loader.skip_every,
            train=True,
            shuffle=True,
            source_type=self.config.data_loader.source_type,
            yaml_label_key=label_keys[self.config.expt_name],
        )

        return train_dataset, val_dataset

    def _log(self, statement):
        """
        Logs a statement in a training log file.

        :param: str statement: a statement to add to the training log file.
        """
        if self.stdout:
            print(statement)

        # Write statement to log file
        with open(self._log_path, "a+") as logf:
            logf.write(statement)
            logf.write("\n")

    def _debug(self, statement):
        """
        Logs a statement in the training debugging file.

        :param: str satatement: a statement to add to the debugging log file.
        """

        if self.stdout:
            print(statement)

        # Write statement to debug file
        with open(self._debug_path, "a+") as logf:
            logf.write(statement)
            logf.write("\n")
