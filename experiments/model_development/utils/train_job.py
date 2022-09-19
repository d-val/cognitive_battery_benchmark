"""
train_job.py: contains the implementation of a model training job and its interface with the config file.
"""

import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from utils.translators import expts

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
    Trains and evaluates model based on a configuration file.
    """
    def __init__(self, config, stdout=True, using_ffcv=False):
        """
        Initialize the job and its parameters.
        """

        # Public training job attributes
        self.config = config
        self.using_ffcv = using_ffcv
        self.cnn_architecture = config.model.cnn_architecture
        self.stdout = stdout
        self.label_translator = expts[config.expt_name]

        # Output set up
        self._start_time = re.sub(r"[^\w\d-]", "_", str(datetime.now()))
        self._out_path = f"output/{self.config.job_name}_{self._start_time}"
        os.makedirs(self._out_path)
        self._log_path = os.path.join(self._out_path, "training.log")
        self._debug_path = os.path.join(self._out_path, "debugging.log")
        self._best_model_path = os.path.join(self._out_path, "model.pt")
        self.config.write_yaml(os.path.join(self._out_path, "config.yaml"))

        data_rng, rng = jax.random.split(rng)
        dataset = train_utils.get_dataset(
        config, data_rng

        self.train_videos, self.train_labels = # TODO !! see https://github.com/deepmind/dmvr/tree/master/examples#creating-and-reading-your-own-dmvr-dataset-using-open-source-tools
        self.test_videos, self.test_labels = # TODO !! see https://github.com/deepmind/dmvr/tree/master/examples#creating-and-reading-your-own-dmvr-dataset-using-open-source-tools

        self.model = model = tf.saved_model.load(self.config.model_dir)
        # TODO ! self.loss_fn = nn.CrossEntropyLoss()
        # TODO ! self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.train_params.lr)

        # Initializing log and log metadata
        self._log(f"Starting Log")
        self.train_losses = []
        self.test_losses = []

    def train(self, evaluate=False):
        """
        Runs the training job by training the model on the training data.

        :param bool evaluate: whether to evaluate the model at each epoch and save the best model.
        :return: training and testing accuracies and losses.
        :rtype: dict["train":tuple(float, float), "test":tuple(float, float)]
        """

        model.fit(self.train_videos, self.train_labels, epochs=self.config.train_params.epochs)

        self._debug("Started training")
        self._log("TRAINING")

    def evaluate(self):
        """
        Evaluates the model on the training and testing datasets.

        :return: training and testing accuracies and losses.
        :rtype: dict["train":tuple(float, float), "test":tuple(float, float)]
        """

        self._debug("\t Checking accuracy on training data")
        train_acc, train_loss = self.model.evaluate(self.train_videos, self.train_labels, verbose=2)

        self._debug("\t Checking accuracy on test data")
        test_acc, test_loss = self.model.evaluate(self.test_videos, self.test_labels, verbose=2)

        return {"train": (train_acc, train_loss), "test":(test_acc, test_loss)}

    def plot(self, show=True, save=True):
        """
        Generates a plot of training and test loss over epochs.

        :param boolean show: whether to show the generated plot
        :param boolean save: whether to save the generated plot
        """
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.test_losses, label="Testing Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Testing Loss vs. Epoch for Model " + self.config.model.name)
        plt.legend()

        if save:
            plt.savefig(os.path.join(self._out_path, "loss.png"))

        if show:
            plt.show()

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

if __name__ == '__main__':
    config = TrainingConfig.from_yaml("config/ModelArchitecture.yaml")
    job = TrainingJob(config=config)
