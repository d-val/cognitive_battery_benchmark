"""
train.py: script to run a training job using config/train.yaml
"""

from utils.train_job import TrainingConfig, TrainingJob
import utils.translators as translators

# First, load the training config
config = TrainingConfig.from_yaml("config/ModelArchitecture.yaml")

# Construct a training job and run it with per-epoch evaluation.
job = TrainingJob(config=config)
job.train(evaluate=True)