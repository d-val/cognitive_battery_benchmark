"""
train.py: script to run a training job using config/train.yaml
"""

import argparse
from utils.train_job import TrainingConfig, TrainingJob


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cnn",
        type=str,
        choices=["resnet18", "resnet34", "alexnet"],
        default=None,
        help="The CNN architecture to use for the vanilla CNN+LSTM model. If provided, overrides config.",
    )
    p.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="The RNG seed to use. If provided, overrides config.",
    )
    p.add_argument(
        "--train_only",
        default=False,
        action="store_true",
        help="add to only run training (no evaluation)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # First, load the training config
    config = TrainingConfig.from_yaml("config/config.yaml")

    if args.cnn is not None:
        config.model.cnn_architecture = args.cnn
    if args.seed is not None:
        config.seed = args.seed

    # Construct a training job and run it with per-epoch evaluation.
    job = TrainingJob(config=config)
    job.train(evaluate=not args.train_only)

    # Generate and save a plot of training and test losses
    job.plot(show=True, save=True)
