from pipeline import TrainModelPipeline
from pipeline import VideoDatasetPipeline
import os
import argparse
import yaml

from model import VideoMAEModel, TimesformerModel, XClipModel


# save your trained model.py checkpoint to wandb
os.environ["WANDB_LOG_MODEL"] = "true"

# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VideoMAEModel on a dataset")
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_config.yaml",
        help="path to model config file",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="dataset_config.yaml",
        help="path to dataset config file",
    )
    args = parser.parse_args()
    return args


# if main

if __name__ == "__main__":
    args = parse_args()

    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    with open(args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    epochs, model_str, batch_size = (
        model_config["epochs"],
        model_config["model"],
        model_config["batch_size"],
    )
    dataset_name, dataset_class_split, dataset_percentage_split = (
        dataset_config["dataset_name"],
        dataset_config["dataset_class_split"],
        dataset_config["dataset_percentage_split"],
    )

    os.environ["WANDB_PROJECT"] = f"{model_str}-{dataset_name}"

    model = eval(model_str)(args.dataset_config)

    dataset = VideoDatasetPipeline(
        args.dataset_config,
        "final_greater_side",
        dataset_class_split=dataset_class_split,
        dataset_percentage_split=dataset_percentage_split,
    )

    dataset.preprocess(model)
    train_pipeline = TrainModelPipeline(model, dataset)

    train_pipeline.train(60, 3)
    train_pipeline.test(3)
