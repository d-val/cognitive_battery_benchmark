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
from utils.translators import expts, label_keys

from utils_motionformer.models.Motionformer.slowfast.config.defaults import get_cfg
from utils_motionformer.models.Motionformer.slowfast.models import build_model
from utils_motionformer.models.Motionformer.slowfast.models import vit_helper
from utils_motionformer.models.Motionformer.slowfast.models.video_model_builder import VisionTransformer
import math

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.cnn_architecture = config.model.cnn_architecture
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
        self.train_loader, self.test_loader = self._get_loaders()
        if ckpt_path:
            self.model = torch.load(ckpt_path)
        else:
            self.defaults_cfg = get_cfg()
            self.cfg_url_name = 'vit_1k'

            self.model = VisionTransformer(self.defaults_cfg)
            vit_helper.load_pretrained(
                                       self.model, cfg=self.defaults_cfg,
                                       in_chans=self.defaults_cfg.VIT.CHANNELS, filter_fn=vit_helper._conv_filter,
                                       strict=False, cfg_url_name=self.cfg_url_name
            )
            if hasattr(self.model, 'st_embed'):
                self.model.st_embed.data[:, 1:, :] = self.model.pos_embed.data[:, 1:, :].repeat(1, self.defaults_cfg.VIT.TEMPORAL_RESOLUTION, 1)
                self.model.st_embed.data[:, 0, :] = self.model.pos_embed.data[:, 0, :]
        
        self.model.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.config.train_params.lr
        )

        # Initializing log and log metadata
        self._log(f"Starting Log, {self.cnn_architecture} + LSTM")

        # Keep count of samples seen in training
        self.train_count = 0

    def train(self, evaluate=False):
        """
        Runs the training job by training the model on the training data.

        :param bool evaluate: whether to evaluate the model at each epoch and save the best model.
        :return: training and testing accuracies and losses.
        :rtype: dict["train":tuple(float, float), "test":tuple(float, float)]
        """

        # Set the model in training state
        self.model.train()

        self._debug("Started training")
        self._log("TRAINING")

        best_loss = float("inf")
        for epoch in range(1, self.config.train_params.epochs + 1):
            for it, (data, targets) in enumerate(self.train_loader):

                # Images are in NHWC, torch works in NCHW
                self._debug(f"Epoch:{epoch}, it:{it}")
                data = torch.permute(data, (0, 1, 4, 2, 3))

                # get data to cuda if possible
                data = data.to(device=device).squeeze(1)
                if self.using_ffcv:
                    targets = targets.to(device=device).squeeze(1)
                else:
                    targets = targets.to(device=device)

                # forward
                prediction = self.model(data)
                loss = self.loss_fn(prediction, targets)

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                self.writer.add_scalar("Loss/train", loss.item(), self.train_count)
                self.train_count += 1

                # gradient descent/optimizer step
                self.optimizer.step()

            if evaluate:
                # Calculate training and testing accuracies and losses for this epoch
                evals = self.evaluate()

                train_acc, train_loss = evals["train"]
                self.writer.add_scalar("Accuracy/train_epoch", train_acc, epoch)
                self.writer.add_scalar("Loss/train_epoch", train_loss, epoch)

                test_acc, test_loss = evals["test"]
                self.writer.add_scalar("Accuracy/test_epoch", test_acc, epoch)
                self.writer.add_scalar("Loss/test_epoch", test_loss, epoch)

                self._log(
                    f"epoch={epoch},train_acc={train_acc:.2f},test_acc={test_acc:.2f},train_loss={train_loss:.2f},test_loss={test_loss:.2f}"
                )
                self.writer.flush()

                # Update best model file if a better model is found.
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(self.model, self._best_model_path)

                if self.config.train_params.save_all_epochs:
                    torch.save(self.model, self._epoch_model_path % epoch)

    def evaluate(self):
        """
        Evaluates the model on the training and testing datasets.

        :return: training and testing accuracies and losses.
        :rtype: dict["train":tuple(float, float), "test":tuple(float, float)]
        """

        self._debug("\t Checking accuracy on training data")
        train_acc, train_loss = self._check_accuracy(self.train_loader)

        self._debug("\t Checking accuracy on test data")
        test_acc, test_loss = self._check_accuracy(self.test_loader)

        return {"train": (train_acc, train_loss), "test": (test_acc, test_loss)}

    def _check_accuracy(self, loader):
        """
        Checks the accuracy and loss of a model on a data loader.

        :param DataLoader loader: a loader of data samples to check the accuracy against.
        :return: the training accuracy and the per-batch loss.
        :rtype: tuple(float, float)
        """
        num_correct, num_samples, running_loss = 0, 0, 0

        # Set the model to evaluation state
        self.model.eval()

        with torch.no_grad():
            for x, y in loader:

                # Pre-process data and correct labels
                x = torch.permute(x, (0, 1, 4, 2, 3))
                x = x.to(device=device).squeeze(1)
                y = y.to(device=device)
                if self.using_ffcv:
                    y = y.squeeze(1)

                # Get model predictions and calculate loss
                scores = self.model(x)
                _, prediction = scores.max(1)
                loss = self.loss_fn(scores, y)

                # Compute accuracy and loss so far
                num_correct += (prediction == y).sum()
                num_samples += prediction.size(0)
                running_loss += loss.item()

            self._debug(
                f"\t Got {num_correct} / {num_samples} with accuracy  \
                {float(num_correct)/float(num_samples)*100:.2f}"
            )
            acc = float(num_correct) / float(num_samples) * 100

        # Reset the model to train state
        self.model.train()

        return acc, running_loss / len(loader)

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

    def _get_loaders(self):
        """
        Creates datasets and data loaders from the current data directory.

        :return: two data loader containing the training data and testing data, respectively.
        :rtype: tuple(DataLoader, DataLoader)
        """
        data_path = self.config.data_loader.data_path
        if self.using_ffcv:
            # Import necessary FFCV defs
            from ffcv.loader import Loader, OrderOption
            from ffcv.transforms import ToTensor
            from ffcv.fields.decoders import IntDecoder, NDArrayDecoder

            # Preprocessing pipeline
            pipelines = {
                "video": [NDArrayDecoder(), ToTensor()],
                "label": [IntDecoder(), ToTensor()],
            }

            # Initialize training and testing data loaders.
            train_loader = Loader(
                data_path,
                batch_size=self.config.data_loader.batch_size,
                num_workers=1,
                order=OrderOption.RANDOM,
                pipelines=pipelines,
            )
            test_loader = train_loader  # TODO: add train/test split for FFCV

        else:
            # Initializing datasets and data-loaders.
            full_dataset = FramesDataset(
                data_path,
                self.label_translator,
                fpv=None,
                skip_every=self.config.data_loader.skip_every,
                train=True,
                shuffle=True,
                source_type=self.config.data_loader.source_type,
                yaml_label_key=label_keys[self.config.expt_name],
            )
            train_size = int(self.config.data_loader.train_split * len(full_dataset))
            test_size = len(full_dataset) - train_size

            # Construct loaders from datasets
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size]
            )
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.config.data_loader.batch_size,
                shuffle=True,
                collate_fn=collate_videos,
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.config.data_loader.batch_size,
                shuffle=True,
                collate_fn=collate_videos,
            )

        return train_loader, test_loader


if __name__ == "__main__":
    config = TrainingConfig.from_yaml("config/ModelArchitecture.yaml")
    job = TrainingJob(config=config)
