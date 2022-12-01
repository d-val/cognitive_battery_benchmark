"""
train.py: script to run a training job using config/train.yaml
"""

from utils.train_job import TrainingConfig, TrainingJob

# First, load the training config
config = TrainingConfig.from_yaml("config/config.yaml")

# Construct a training job and run it with per-epoch evaluation.
job = TrainingJob(config=config)
job.train(evaluate=True)

# Generate and save a plot of training and test losses
job.plot(show=True, save=True)
