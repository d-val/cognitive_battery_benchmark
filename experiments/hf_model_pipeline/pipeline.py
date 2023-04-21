import glob

import numpy as np
import yaml
import pytorchvideo
import torch
import evaluate
from torch.utils.data import random_split
from torchvision.transforms import (
    Compose,
    Lambda,
    Resize,
)
from pytorchvideo.data import LabeledVideoDataset

from transformers import TrainingArguments, Trainer

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)


def split_list_by_percentages(input_list, percentages):
    np.random.shuffle(input_list)
    total_len = len(input_list)
    sublists = []
    start = 0

    if len(percentages) == 0:
        return [input_list]

    for percentage in percentages:
        end = start + int(total_len * percentage)
        sublists.append(input_list[start:end])
        start = end

    return sublists


class TrainModelPipeline:
    def __init__(self, preprocessor, model, video_dataset, postprocessor=None):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor
        self.datasets = video_dataset.datasets

    def train(
            self,
            num_epochs,
            batch_size,
            learning_rate=5e-5,
            optimized_metric="accuracy",
            new_model_name="fine_tuned_model",
    ):
        accuracy = evaluate.load(optimized_metric)

        dataset = self.datasets[0]
        train_dataset, val_dataset = dataset[0], dataset[1]

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return accuracy.compute(
                predictions=predictions, references=eval_pred.label_ids
            )

        def collate_fn(examples):
            # permute to (num_frames, num_channels, height, width)
            pixel_values = torch.stack(
                [example["video"].permute(1, 0, 2, 3) for example in examples]
            )
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        args = TrainingArguments(
            new_model_name,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model=optimized_metric,
            push_to_hub=False,
            max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        )

        trainer = Trainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.preprocessor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )

        train_results = trainer.train()
        return train_results

    def test(
            self,
            batch_size,
            optimized_metric="accuracy",
    ):
        test_dataset = self.datasets[1][0]
        accuracy = evaluate.load(optimized_metric)

        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return accuracy.compute(
                predictions=predictions, references=eval_pred.label_ids
            )

        def collate_fn(examples):
            # permute to (num_frames, num_channels, height, width)
            pixel_values = torch.stack(
                [example["video"].permute(1, 0, 2, 3) for example in examples]
            )
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        args = TrainingArguments(
            "testing_model",
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_eval_batch_size=batch_size,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model=optimized_metric,
            push_to_hub=False,
        )

        trainer = Trainer(
            self.model,
            args,
            eval_dataset=test_dataset,
            tokenizer=self.preprocessor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )

        test_results = trainer.evaluate()
        return test_results



class VideoDatasetPipeline:
    def __init__(
            self,
            path,
            label_arg,
            video_ext="mp4",
            dataset_class_split=None,
            dataset_percentage_split=None,
    ):
        if dataset_percentage_split is None:
            dataset_percentage_split = [[0.75, 0, 25], []]
        if dataset_class_split is None:
            dataset_class_split = [["1", "2", "3", "4"], ["5", "6"]]
        self.datasets = None
        self.resize_to = None
        sub_folders = glob.glob(path + "/*/")
        videos_w_labels = {}

        classes = []
        assert sub_folders != [], "No subfolders found in path"
        for _, folder in enumerate(sub_folders):
            sub_videos_w_labels = []
            runs = glob.glob(folder + "/*")
            for run in runs:
                video = glob.glob(run + f"/*.{video_ext}")
                if len(video) != 1:
                    raise Exception(f"Found {len(video)} videos in {run}")
                video = video[0]
                exp_file = run + "/machine_readable/experiment_stats.yaml"
                with open(exp_file, "r") as f:
                    exp = yaml.safe_load(f)
                    label = str(exp[label_arg])
                    if label not in classes:
                        classes.append(label)

                sub_videos_w_labels.append((video, {"label": classes.index(label)}))
            videos_w_labels[folder.split('/')[-2]] = sub_videos_w_labels

        # unroll dict and torch.utils.data.random_split
        # assert that all items in dataset_split are floats

        # unroll dict and torch.utils.data.random_split

        # assert that all items in dataset_split are lists of float
        assert all(
            [isinstance(x, list) for x in dataset_percentage_split]
        ), "dataset_split must be a list of lists"
        assert all(
            [all([isinstance(y, float) for y in x]) for x in dataset_percentage_split]
        ), "dataset_split must be a list of lists of floats"

        # assert that all items in dataset_class_splits are lists of strings
        assert all(
            [isinstance(x, list) for x in dataset_class_split]
        ), "dataset_split must be a list of lists"
        assert all(
            [all([isinstance(y, str) for y in x]) for x in dataset_class_split]
        ), "dataset_split must be a list of lists of strings"

        self.ds_splits = []
        for percentage_split, class_split in zip(dataset_percentage_split, dataset_class_split):
            split_videos = []
            for label in class_split:
                split_videos.extend(videos_w_labels[label])
            partitioned_videos = split_list_by_percentages(split_videos,
                                                           percentage_split)
            self.ds_splits.append(partitioned_videos)

        self.label2id = {label: i for i, label in enumerate(classes)}
        self.id2label = {i: label for i, label in enumerate(classes)}
        print("Loaded dataset")

    def preprocess(
            self,
            preprocessor,
            model
    ):
        self.mean = preprocessor.image_mean
        self.std = preprocessor.image_std
        if "shortest_edge" in preprocessor.size:
            height = width = preprocessor.size["shortest_edge"]
        else:
            height = preprocessor.size["height"]
            width = preprocessor.size["width"]
        self.resize_to = (height, width)

        num_frames_to_sample = model.config.num_frames
        sample_rate = 4
        fps = 30
        clip_duration = num_frames_to_sample * sample_rate / fps
        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            Resize(self.resize_to),
                        ]
                    ),
                ),
            ]
        )

        self.datasets = [
            [LabeledVideoDataset(
                sub_ds_split,
                clip_sampler=pytorchvideo.data.make_clip_sampler(
                    "uniform", clip_duration
                ),
                video_sampler=torch.utils.data.sampler.RandomSampler,
                transform=transform,
                decoder="pyav",
            ) for sub_ds_split in ds_split]
            for ds_split in self.ds_splits
        ]

        print("Preprocessed dataset")
