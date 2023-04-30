from transformers import AutoFeatureExtractor, AutoModelForVideoClassification
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification


def VideoMAEModel():
    def __init__(self, dataset):
        model_ckpt = "MCG-NJU/videomae-base"
        self.preprocessor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            ignore_mismatched_sizes=True,
        )


def TimesformerModel():
    def __init__(self, dataset):
        model_ckpt = "facebook/timesformer-base-finetuned-ssv2"
        self.preprocessor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        self.model = AutoModelForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            ignore_mismatched_sizes=True,
        )


def XClipModel():
    def __init__(self, dataset):
        model_ckpt = "microsoft/xclip-base-patch32"
        self.preprocessor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        self.model = AutoModelForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            ignore_mismatched_sizes=True,
        )
