import os
from collections import OrderedDict

import pandas as pd
from hf_utils import (
    find_best_ckpts,
    load_model_and_dataset,
    parse_args_evaluate,
    seed_everything,
)
from pipeline import MultilabelTrainer, custom_video_collate_fn, gen_compute_metrics
from transformers import Trainer, TrainingArguments


def run_inference_on_model_dataset(
    model_name: str, dataset_name: str, ckpt_path: str, split: int
):
    model, video_dataset, is_multilabel, is_rnn_mode = load_model_and_dataset(
        model_name, dataset_name, ckpt_path, split
    )

    preprocessor = model.preprocessor
    video_dataset.preprocess(model, is_rnn_mode)

    dataset = video_dataset.datasets[1][0]
    output_dir = f"inference_output/{model_name}/{dataset_name}/{split}"

    trainer_type = MultilabelTrainer if is_multilabel else Trainer
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        bf16=False,
        fp16=True,
    )
    trainer = trainer_type(
        model=model.model,
        args=trainer_args,
        tokenizer=preprocessor,
        compute_metrics=gen_compute_metrics("accuracy", is_multilabel=is_multilabel),
        data_collator=lambda x: custom_video_collate_fn(x, is_multilabel, is_rnn_mode),
    )
    results = OrderedDict(
        {
            "model": model_name,
            "dataset": dataset_name,
            "split": split,
        }
    )

    eval_results = trainer.predict(dataset)
    results.update(eval_results.metrics)
    trainer.log(eval_results.metrics)
    print("Test Results:\n", eval_results.metrics)

    inference_csv = "results.csv"
    inference_csv = "tmp.csv"
    new_df = pd.DataFrame([results])

    if os.path.isfile(inference_csv):
        with open(inference_csv, "a+"):
            existing_df = pd.read_csv(inference_csv)
            df = pd.concat([existing_df, new_df], ignore_index=True)
            df.to_csv(inference_csv, index=False)
    else:
        with open(inference_csv, "a+"):
            new_df.to_csv(inference_csv, index=False)


if __name__ == "__main__":
    args = parse_args_evaluate()
    seed_everything()

    model_name = args.model_name
    dataset_name = args.dataset_name
    split = args.split

    ckpt_path = find_best_ckpts(model=model_name, dataset=f"{dataset_name}_data")
    print("Loading ckpt:", ckpt_path)
    run_inference_on_model_dataset(model_name, dataset_name, ckpt_path, split)
