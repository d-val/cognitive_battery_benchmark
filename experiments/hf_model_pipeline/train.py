from pipeline import TrainModelPipeline
from pipeline import VideoDatasetPipeline
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import os

os.environ["WANDB_PROJECT"]="MAE-RelativeNumbers"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


dataset = VideoDatasetPipeline("./RelativeNumbers", "final_greater_side", dataset_class_split=[["1", "2", "3", "4"], ["5", "6"]], dataset_percentage_split=[[0.75, 0.25], []])
model_ckpt = "MCG-NJU/videomae-base"
preprocessor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=dataset.label2id,
    id2label=dataset.id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
dataset.preprocess(preprocessor, model)
train_pipeline = TrainModelPipeline(preprocessor, model, dataset)

train_pipeline.train(1, 3)
train_pipeline.test(3)