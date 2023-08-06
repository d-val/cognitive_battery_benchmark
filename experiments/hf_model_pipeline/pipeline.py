import glob
import math
import time
from functools import partial

import numpy as np
import pytorchvideo
import sklearn.metrics as skm
import torch
import yaml
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipInfo, ClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda, Resize
from transformers import Trainer, TrainerCallback, TrainingArguments


class TimeConstraintCallback(TrainerCallback):
    def __init__(self, max_training_time_seconds, test_fn):
        super().__init__()
        self.max_training_time_seconds = max_training_time_seconds
        self.training_start_time = None
        self.test_fn = test_fn

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.training_start_time
        if elapsed_time > self.max_training_time_seconds:
            print("Stopping training")
            print("Running test suite")
            self.test_fn()
            control.should_training_stop = True
        # print(f"Step took {elapsed_time:.2f}")

    def on_train_end(self, args, state, control, **kwargs):
        print("Running test suite")
        self.test_fn()


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        assert (
            len(labels.shape) == 2
        ), f"Labels should be 2D, but got shape {labels.shape}"
        outputs = model(**inputs)
        logits = outputs.get("logits")
        assert (
            logits.shape == labels.shape
        ), f"Logits and labels should be the same shape, but got {logits.shape} and {labels.shape}"
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


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


def gen_compute_metrics(opt_metric="accuracy", threshold=0.5, is_multilabel=False):
    def compute_metrics_function(eval_pred):
        labels = eval_pred.label_ids
        # If the task is multilabel, predict a class if the model gives it a probability greater than the threshold
        if is_multilabel:
            predictions = (eval_pred.predictions > threshold).astype(int).flatten()
            labels = labels.flatten()
        # If the task is not multilabel, predict the class with the highest probability
        else:
            predictions = np.argmax(eval_pred.predictions, axis=1)

        metrics = {
            "accuracy": skm.accuracy_score(labels, predictions),
            "mae": skm.mean_absolute_error(labels, predictions),
            "precision": skm.precision_score(labels, predictions, average="macro"),
            "recall": skm.recall_score(labels, predictions, average="macro"),
            "f1": skm.f1_score(labels, predictions, average="macro"),
        }

        cm = skm.multilabel_confusion_matrix(labels, predictions)

        TN = cm[:, 0, 0].sum()
        FN = cm[:, 1, 0].sum()
        TP = cm[:, 1, 1].sum()
        FP = cm[:, 0, 1].sum()

        metrics.update({"TP": TP, "FP": FP, "TN": TN, "FN": FN})

        return metrics

    return compute_metrics_function


def custom_video_collate_fn(examples, multilabel: bool, rnn_mode: bool):
    # permute to (num_frames, num_channels, height, width)
    if not rnn_mode:
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
    else:
        fps = 30
        n_frames_sec = 4
        frame_gap = fps // n_frames_sec
        channels, _, height, width = examples[0]["video"].shape
        max_frames = max(
            [math.ceil(example["video"].shape[1] / frame_gap) for example in examples]
        )

        # Pad all sequences to have max_frames number of frames
        pixel_values = torch.stack(
            [
                torch.cat(
                    [
                        example["video"][:, ::frame_gap, :, :],
                        torch.zeros(
                            channels,
                            max_frames
                            - math.ceil(example["video"].shape[1] / frame_gap),
                            height,
                            width,
                        ),
                    ],
                    dim=1,
                )
                for example in examples
            ]
        )

        # permute to (num_frames, num_channels, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

    if not multilabel:
        labels = torch.tensor([example["label"] for example in examples])
    else:
        list_labels = [example["label"] for example in examples]
        labels = torch.stack(list_labels)

    return {"pixel_values": pixel_values, "labels": labels}


class TrainModelPipeline:
    def __init__(
        self,
        args,
        model_class,
        video_dataset,
        postprocessor=None,
        multilabel=False,
        rnn_mode=False,
    ):
        self.preprocessor = model_class.preprocessor
        self.model = model_class.model
        self.postprocessor = postprocessor
        self.datasets = video_dataset.datasets
        self.multilabel = multilabel
        self.rnn_mode = rnn_mode

        self.learning_rate = args.lr
        self.warmup_ratio = args.warmup_ratio
        self.run_name = f"{model_class.name}/{video_dataset.name}/lr{self.learning_rate}_wr{self.warmup_ratio}"

    def train(
        self, num_epochs, batch_size, optimized_metric="accuracy", time_limit=None
    ):
        new_model_name = f"output/{self.run_name}"

        dataset = self.datasets[0]
        train_dataset, val_dataset = dataset[0], dataset[1]
        callback_list = []
        if time_limit is not None:
            test_with_batch_size = partial(self.test, batch_size=batch_size)
            callback_list.append(
                TimeConstraintCallback(
                    max_training_time_seconds=time_limit - 600,
                    test_fn=test_with_batch_size,
                )
            )

        args = TrainingArguments(
            new_model_name,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=5,
            logging_strategy="epoch",
            learning_rate=self.learning_rate,
            load_best_model_at_end=True,
            metric_for_best_model=optimized_metric,
            num_train_epochs=num_epochs,
            report_to="tensorboard",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            lr_scheduler_type="linear",
            warmup_ratio=self.warmup_ratio,
            optim="adamw_torch_fused",
            bf16=False,
            fp16=True,
        )

        TrainerType = None
        if self.multilabel:
            TrainerType = MultilabelTrainer
        else:
            TrainerType = Trainer

        self.trainer = TrainerType(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.preprocessor,
            compute_metrics=gen_compute_metrics(
                optimized_metric, is_multilabel=self.multilabel
            ),
            data_collator=lambda x: custom_video_collate_fn(
                x, self.multilabel, self.rnn_mode
            ),
            callbacks=callback_list,
        )

        train_results = self.trainer.train()
        return train_results

    def test(self, batch_size, optimized_metric="accuracy"):
        test_dataset = self.datasets[1][0]

        # Evaluate the test dataset
        eval_results = self.trainer.predict(test_dataset)
        self.trainer.log(eval_results.metrics)

        # Log the evaluation results
        print({"test_results": eval_results.metrics})


class VideoDatasetPipeline:
    def __init__(
        self,
        name,
        path,
        label_arg,
        video_ext="mp4",
        dataset_class_split=None,
        dataset_percentage_split=None,
        multi_label=False,
        entire_video=False,
    ):
        self.name = name
        self.std = None
        self.mean = None
        self.entire_video = entire_video
        if dataset_percentage_split is None:
            dataset_percentage_split = [[0.75, 0, 25], []]
        if dataset_class_split is None:
            dataset_class_split = [["1", "2", "3", "4"], ["5", "6"]]
        self.datasets = None
        self.resize_to = None
        sub_folders = glob.glob(path + "/*/")
        videos_w_labels = {}

        classes = set()
        assert sub_folders != [], "No subfolders found in path"
        for _, folder in enumerate(sub_folders):
            runs = glob.glob(folder + "/*")
            for run in runs:
                video = glob.glob(run + f"/*.{video_ext}")
                if len(video) != 1:
                    raise Exception(f"Found {len(video)} videos in {run}")
                exp_file = run + "/machine_readable/experiment_stats.yaml"
                with open(exp_file, "r") as f:
                    exp = yaml.load(f, Loader=yaml.Loader)
                    if not multi_label:
                        label = str(exp[label_arg])
                        if label not in classes:
                            classes.add(label)
                    else:
                        classes.update([str(x) for x in exp[label_arg]])

        classes = list(classes)
        classes.sort()
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
                    try:
                        exp = yaml.load(f, Loader=yaml.Loader)
                        if not multi_label:
                            sub_videos_w_labels.append(
                                (video, {"label": classes.index(str(exp[label_arg]))})
                            )
                        else:
                            full_labels = []
                            for sub_label in exp[label_arg]:
                                full_labels.append(classes.index(str(sub_label)))

                            num_classes = len(classes)
                            labels_tensor = torch.zeros(
                                (num_classes,), dtype=torch.long
                            )

                            for label in full_labels:
                                labels_tensor[label] = 1

                            sub_videos_w_labels.append(
                                (video, {"label": labels_tensor})
                            )
                    except Exception as E:
                        print(f"Failed on {exp_file}\n")
                        print(E)

            videos_w_labels[folder.split("/")[-2]] = sub_videos_w_labels
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
        for percentage_split, class_split in zip(
            dataset_percentage_split, dataset_class_split
        ):
            split_videos = []
            for label in class_split:
                split_videos.extend(videos_w_labels[label])
            partitioned_videos = split_list_by_percentages(
                split_videos, percentage_split
            )
            self.ds_splits.append(partitioned_videos)

        self.label2id = {label: i for i, label in enumerate(classes)}
        self.id2label = {i: label for i, label in enumerate(classes)}
        print("Loaded dataset")

    def preprocess(self, model_class, rnn_bool=False):
        preprocessor = model_class.preprocessor
        model = model_class.model
        self.mean = preprocessor.image_mean
        self.std = preprocessor.image_std
        if "shortest_edge" in preprocessor.size:
            height = width = preprocessor.size["shortest_edge"]
        else:
            height = preprocessor.size["height"]
            width = preprocessor.size["width"]
        self.resize_to = (height, width)
        if not rnn_bool:
            num_frames_to_sample = model.config.num_frames
            transform_compose = Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(self.mean, self.std),
                    Resize(self.resize_to, antialias=True),
                ]
            )
        else:
            transform_compose = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    Normalize(self.mean, self.std),
                    Resize(self.resize_to, antialias=True),
                ]
            )

        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=transform_compose,
                ),
            ]
        )

        clip_sampler = None

        if not self.entire_video:
            sample_rate = 11
            fps = 30
            clip_duration = num_frames_to_sample * sample_rate / fps
            clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", clip_duration)
        else:
            clip_sampler = EntireVideoSampler()
        self.datasets = [
            [
                CustomVideoDataset(
                    sub_ds_split,
                    clip_sampler=clip_sampler,
                    video_sampler=torch.utils.data.sampler.SequentialSampler,
                    transform=transform,
                    decoder="pyav",
                )
                for sub_ds_split in ds_split
            ]
            for ds_split in self.ds_splits
        ]

        print(f"Preprocessed dataset, entire video: {self.entire_video}")


class EntireVideoSampler(ClipSampler):
    """
    This class returns the entire video as a single clip.
    """

    def __init__(self) -> None:
        super().__init__(0)  # clip_duration is 0 since we want the entire video

    def __call__(self, last_clip_time, video_duration, annotation):
        """
        Args:
            last_clip_time (float): Not used for EntireVideoSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled.
            annotation (Dict): Not used by this sampler.
        Returns:
            a named-tuple `ClipInfo`: includes the clip information of (clip_start_time,
                clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
                is_last_clip is True after clips_per_video clips have been sampled or the end
                of the video is reached.
        """

        clip_start_sec = 0
        clip_index = 0
        aug_index = 0
        is_last_clip = True  # True because there's only one clip, the entire video

        return ClipInfo(
            clip_start_sec,
            video_duration,
            clip_index,
            aug_index,
            is_last_clip,
        )


class CustomVideoDataset(LabeledVideoDataset):
    def __len__(self):
        return self.num_videos
