import argparse
import json
import os
import random
from glob import glob

import numpy as np
import torch
import transformers
import yaml
from model import *
from pipeline import VideoDatasetPipeline


DATASET_CONFIGS = {
    "addition": "configs/dataset/addition.yaml",
    "gravity": "configs/dataset/gravity.yaml",
    "relative": "configs/dataset/relative.yaml",
    "rotation": "configs/dataset/rotation.yaml",
    "shape": "configs/dataset/shape.yaml",
    "swap": "configs/dataset/swap.yaml",
}

MODEL_CONFIGS = {
    "densenet": "configs/model/DensenetLSTM.yaml",
    "resnet18": "configs/model/Resnet18LSTM.yaml",
    "resnet50": "configs/model/Resnet50LSTM.yaml",
    "timesformer": "configs/model/Timesformer.yaml",
    "vit-b16": "configs/model/ViTB16LSTM.yaml",
    "videomae": "configs/model/VideoMAE.yaml",
    "xclip": "configs/model/XClip.yaml",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    transformers.set_seed(seed)


def parse_args_train():
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
    parser.add_argument(
        "--time_limit_in_minutes",
        type=int,
        default=None,
        help="how long to train job",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="training learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0,
        help="optimizer warmup ratio",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="number of training epochs (overrides config)",
    )

    args = parser.parse_args()
    return args


def parse_args_evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


def load_model_and_dataset(
    model_name: str, dataset_name: str, ckpt_path: str, split: str
):
    with open(DATASET_CONFIGS[dataset_name], "r") as f:
        dataset_config = yaml.safe_load(f)

    with open(MODEL_CONFIGS[model_name], "r") as f:
        model_config = yaml.safe_load(f)

    is_multilabel: bool = dataset_config.get("multi_label", False)
    is_rnn_mode: bool = model_config.get("rnn_bool", False)

    folder = f"/home/gridsan/kshehada/data/{dataset_config['dataset_name']}/videos"

    dataset_splits = [
        i for j in dataset_config["dataset_class_split"] for i in j if i != split
    ]
    dataset_splits = [dataset_splits, [split]]

    video_dataset = VideoDatasetPipeline(
        dataset_name,
        folder,
        dataset_config.get("dataset_label", "final_greater_side"),
        dataset_class_split=dataset_splits,
        dataset_percentage_split=dataset_config["dataset_percentage_split"],
        multi_label=is_multilabel,
        entire_video=dataset_config.get("entire_video", False),
    )
    try:
        model = eval(model_config["model"])(video_dataset)
        weights = torch.load(f"{ckpt_path}/pytorch_model.bin")
        model.model.load_state_dict(weights)
    except RuntimeError:
        model = eval(model_config["model"])(video_dataset, is_old=True)
        weights = torch.load(f"{ckpt_path}/pytorch_model.bin")
        model.model.load_state_dict(weights)
    # model = eval(model_config["model"])(video_dataset)

    model.model = model.model.to(device)

    return model, video_dataset, is_multilabel, is_rnn_mode


def find_best_ckpts(model_name: str, dataset_name: str):
    m, d = model_name, dataset_name

    runs = glob(f"output/{m}/{d}/**")
    best_ckpt = None
    best_loss = np.inf
    for r in runs:
        run_ckpts = []
        for ckpt_dir in os.listdir(r):
            if not ckpt_dir.startswith("checkpoint"):
                continue
            ckpt = int(ckpt_dir.split("-")[-1])
            run_ckpts.append(ckpt)

        max_ckpt = max(run_ckpts)
        max_config_path = f"{r}/checkpoint-{max_ckpt}/trainer_state.json"

        with open(max_config_path) as f:
            trainer_log = json.load(f)["log_history"]

        ckpts_logs = {}
        for l in trainer_log:
            if (int(l["step"]) in run_ckpts) and ("eval_loss" in l):
                ckpts_logs[l["step"]] = l["eval_loss"]

        best_run_ckpt = min(ckpts_logs, key=lambda x: ckpts_logs[x])
        best_run_loss = ckpts_logs[best_run_ckpt]

        if best_run_loss < best_loss:
            best_ckpt = f"{r}/checkpoint-{best_run_ckpt}"
            best_loss = best_run_loss

    return best_ckpt
