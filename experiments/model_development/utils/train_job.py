"""
train_job.py: contains the implementation of a model training job and its interface with the config file.
"""

import os
import random
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.functional as mf
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.framesdata import FramesDataset, collate_videos
from utils.model import CNNLSTM
from utils.translators import expts, label_keys

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
        self.metrics = self._get_metrics()

        # Set the random seed, if provided
        if hasattr(self.config, "seed"):
            self._seed_everything(self.config.seed)

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
            self.model = CNNLSTM(
                config.model.lstm_hidden_size,
                config.model.lstm_num_layers,
                config.model.num_classes,
                cnn_architecture=self.cnn_architecture,
                pretrained=True,
            )
        self.model.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        if self.config.train_params.optimizer.lower() == "adam":
            optimizer = optim.Adam
        else:
            optimizer = optim.SGD
        self.optimizer = optimizer(self.model.parameters(), lr=self.config.train_params.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.5)

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
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch}]")
            for data, targets in pbar:
                # Images are in NHWC, torch works in NCHW
                # self._debug(f"Epoch:{epoch}, it:{it}")
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
                nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                self.optimizer.step()
                self.scheduler.step()
                
            if evaluate:
                # Calculate training and testing accuracies and losses for this epoch
                evals = self.evaluate()
                log_statement = f"epoch={epoch}"
                
                train_metrics = evals["train"]
                for m in train_metrics:
                    self.writer.add_scalar(f"{m}/train_epoch", train_metrics[m], epoch)
                    log_statement += f",train_{m}={train_metrics[m]:.4f}"

                test_metrics = evals["test"]
                for m in test_metrics:
                    self.writer.add_scalar(f"{m}/test_epoch", test_metrics[m], epoch)
                    log_statement += f",test_{m}={test_metrics[m]:.4f}"
                
                self._log(log_statement)
                self.writer.flush()

                # Update best model file if a better model is found.
                if test_metrics["loss"] < best_loss:
                    best_loss = test_metrics["loss"]
                    torch.save(self.model, self._best_model_path)

                if self.config.train_params.save_all_epochs:
                    torch.save(self.model, self._epoch_model_path % epoch)

    def evaluate(self):
        """
        Evaluates the model on the training and testing datasets.

        :return: training and testing accuracies and losses.
        :rtype: dict["train":tuple(float, float), "test":tuple(float, float)]
        """

        self._debug("\t Checking metrics on training data")
        train_metrics = self._compute_metrics(self.train_loader)

        self._debug("\t Checking metrics on test data")
        test_metrics = self._compute_metrics(self.test_loader)

        return {"train": train_metrics, "test": test_metrics}

    def _compute_metrics(self, loader):
        """
        Checks the accuracy and loss of a model on a data loader.

        :param DataLoader loader: a loader of data samples to check the accuracy against.
        :return: the training accuracy and the per-batch loss.
        :rtype: tuple(float, float)
        """
        num_correct, num_samples, running_loss = 0, 0, 0

        # Set the model to evaluation state
        self.model.eval()

        outputs, targets = [], []
        
        with torch.no_grad():
            for x, y in tqdm(loader):
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
                
                outputs.append(scores.cpu())
                targets.append(y.cpu())

                # Compute accuracy and loss so far
                num_correct += (prediction == y).sum()
                num_samples += prediction.size(0)
                running_loss += loss.item()

            self._debug(
                f"\t Got {num_correct} / {num_samples} with accuracy  \
                {float(num_correct)/float(num_samples)*100:.2f}"
            )
            acc = float(num_correct) / float(num_samples) * 100
            eval_metrics = {"acc": acc, "loss": running_loss / len(loader)}
            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
            for m in self.metrics:
                eval_metrics[m] = self.metrics[m](outputs, targets).item()
            
            for m in eval_metrics:
                self._debug(f"\t {m}: \t {eval_metrics[m]:.4}")
                
        # Reset the model to train state
        self.model.train()

        return eval_metrics

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
            from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
            from ffcv.loader import Loader, OrderOption
            from ffcv.transforms import ToTensor

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

    def _get_metrics(self):
        metrics = {}
        
        for k in {1, 3, 5}:
            metrics[f"top{k}_acc"] = lambda prediction, targets, k=k: mf.accuracy(
                prediction, targets, task="multiclass", num_classes=8, top_k=k
            )

        for m in {mf.precision, mf.f1_score, mf.recall, mf.auroc}:
            metrics[m.__name__] = lambda prediction, targets: m(
            prediction, targets, task="multiclass", num_classes=8
        )
        metrics["mae"] = lambda prediction, targets: mf.mean_absolute_error(prediction.max(1)[1], targets)
        return metrics

    def _seed_everything(self, seed: int):
        """
        Sets the RNG seed for numpy, pytorch, and Python's random library.

        :param: int seed: the RNG seed to use
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    config = TrainingConfig.from_yaml("config/ModelArchitecture.yaml")
    job = TrainingJob(config=config)
