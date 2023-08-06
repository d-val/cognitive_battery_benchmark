from pipeline import TrainModelPipeline
from pipeline import VideoDatasetPipeline
import yaml

from model import *
from hf_utils import seed_everything, parse_args_train

if __name__ == "__main__":
    args = parse_args_train()
    seed_everything()

    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    with open(args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    epochs, model_str, batch_size, rnn_bool = (
        model_config["epochs"],
        model_config["model"],
        model_config["batch_size"],
        model_config.get("rnn_bool", False),
    )
    (
        dataset_name,
        dataset_class_split,
        dataset_percentage_split,
        dataset_label,
        multi_label,
        entire_video,
    ) = (
        dataset_config["dataset_name"],
        dataset_config["dataset_class_split"],
        dataset_config["dataset_percentage_split"],
        dataset_config.get("dataset_label", "final_greater_side"),
        dataset_config.get("multi_label", False),
        dataset_config.get("entire_video", False),
    )

    if args.num_epochs:
        epochs = args.num_epochs

    time_limit = args.time_limit_in_minutes
    print("Args:", args)
    print(f"Time limit given: {time_limit}")
    if time_limit is not None:
        time_limit *= 60

    folder = f"/home/gridsan/kshehada/data/{dataset_name}/videos"
    print(folder)
    dataset = VideoDatasetPipeline(
        dataset_name,
        folder,
        dataset_label,
        dataset_class_split=dataset_class_split,
        dataset_percentage_split=dataset_percentage_split,
        multi_label=multi_label,
        entire_video=entire_video,
    )
    model = eval(model_str)(dataset)

    dataset.preprocess(model, rnn_bool)
    train_pipeline = TrainModelPipeline(
        args, model, dataset, multilabel=multi_label, rnn_mode=rnn_bool
    )

    train_pipeline.train(epochs, batch_size, time_limit=time_limit)
    train_pipeline.test(batch_size)
