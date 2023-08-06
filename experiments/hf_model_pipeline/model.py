from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForVideoClassification,
    AutoProcessor,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    XCLIPModel,
)
from transformers.modeling_outputs import ImageClassifierOutput


class VideoMAE:
    def __init__(self, dataset):
        model_ckpt = "models/videomae-base"
        self.name = "videomae"
        self.preprocessor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            ignore_mismatched_sizes=True,
        )


class Timesformer:
    def __init__(self, dataset):
        model_ckpt = "models/timesformer-base-finetuned-ssv2"
        self.name = "timesformer"
        self.preprocessor = AutoFeatureExtractor.from_pretrained(model_ckpt)
        self.model = AutoModelForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            ignore_mismatched_sizes=True,
        )


class XCLIPModelForClassification(nn.Module):
    def __init__(
        self,
        model_ckpt,
        label2id,
        id2label,
        ignore_mismatched_sizes,
        problem_type=None,
    ):
        super(XCLIPModelForClassification, self).__init__()

        self.xclip = XCLIPModel.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        self.config = self.xclip.config.vision_config
        num_classes = len(label2id)
        hidden_size = self.config.mit_hidden_size

        self.classifier = nn.Linear(hidden_size, num_classes)

        self.fc_norm = nn.LayerNorm(hidden_size)
        self.classifier = (
            nn.Linear(hidden_size, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.problem_type = problem_type

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        features = self.xclip.get_video_features(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        features = self.fc_norm(features)
        logits = self.classifier(features)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            # print(f"RNN Loss: {loss}")
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


class XClip:
    def __init__(self, dataset):
        model_ckpt = "models/xclip-base-patch32"
        self.name = "xclip"
        self.preprocessor = AutoProcessor.from_pretrained(model_ckpt).current_processor
        self.model = XCLIPModelForClassification(
            model_ckpt,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            ignore_mismatched_sizes=True,
        )


class ImageEmbedder:
    def __init__(self, model_name, frozen=True, hs_mean_dim=[2, 3]):
        self.preprocessor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.name = model_name

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.hs_mean_dim = hs_mean_dim

        if frozen:
            self.model.eval()
        else:
            self.model.train()

    def get_embeddings(self, inputs):
        # print(f'Input size: {inputs.size()}')
        outputs = self.model(inputs)
        embeddings = outputs.last_hidden_state.mean(dim=self.hs_mean_dim)
        # print(f'Embedder size: {embeddings.size()}')
        return embeddings  # mean-pooling to get fixed-size vector


class FrameRNN(nn.Module):
    def __init__(
        self,
        embedder,
        rnn_num_layers,
        rnn_hidden_size,
        num_classes,
        embed_val=None,
        multi_label=False,
        is_old=False,
    ):
        super().__init__()
        self.multi_label = multi_label
        self.embedder = embedder
        if embed_val is not None:
            self.rnn = nn.LSTM(
                num_layers=rnn_num_layers,
                input_size=embedder.model.config[embed_val],
                hidden_size=rnn_hidden_size,
                batch_first=True,
            )
        else:
            if hasattr(embedder.model.config, "hidden_size"):
                self.rnn = nn.LSTM(
                    input_size=embedder.model.config.hidden_size,
                    hidden_size=rnn_hidden_size,
                    batch_first=True,
                )
            else:
                self.rnn = nn.LSTM(
                    input_size=embedder.model.config.hidden_sizes[-1],
                    hidden_size=rnn_hidden_size,
                    batch_first=True,
                )
        self.fc_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, pixel_values, labels=None):
        # pixel_values is expected to be a 5D tensor of shape (batch_size, num_frames, num_channels, height, width)
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.contiguous().view(
            batch_size * num_frames, num_channels, height, width
        )

        frame_embeddings = self.embedder.get_embeddings(pixel_values)

        # Reshape the tensor to shape (batch_size, num_frames, -1)
        frame_embeddings = frame_embeddings.view(batch_size, num_frames, -1)

        output, _ = self.rnn(frame_embeddings)

        output = self.fc_dropout(output)
        output = self.fc(output[:, -1, :])  # take the output from the last RNN state
        # print(f"Output RNN: {output}")
        # print(f"Labels: {labels}")
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, labels)
            # print(f"RNN Loss: {loss}")
            return {"loss": loss, "logits": output}

        return {"logits": output}


class Resnet18LSTM:
    def __init__(self, dataset, is_old=False):
        self.name = "resnet18"
        embedder = ImageEmbedder("models/resnet-18", frozen=False)
        self.preprocessor = embedder.preprocessor
        hidden_size = 128 if is_old else 512
        self.model = FrameRNN(
            embedder, 2, hidden_size, len(dataset.label2id.keys()), is_old=is_old
        )


class Resnet50LSTM:
    def __init__(self, dataset, is_old=False):
        self.name = "resnet50"
        embedder = ImageEmbedder("models/resnet-50", frozen=False)
        self.preprocessor = embedder.preprocessor
        hidden_size = 128 if is_old else 512
        self.model = FrameRNN(
            embedder, 2, hidden_size, len(dataset.label2id.keys()), is_old=is_old
        )


class DensenetLSTM:
    def __init__(self, dataset, is_old=False):
        self.name = "densenet"
        embedder = ImageEmbedder("densenet121.ra_in1k", frozen=False)
        self.preprocessor = embedder.preprocessor
        hidden_size = 128 if is_old else 512
        self.model = FrameRNN(
            embedder, 2, hidden_size, len(dataset.label2id.keys()), is_old=is_old
        )


class ViTB16LSTM:
    def __init__(self, dataset, is_old=False):
        self.name = "vit-b16"
        embedder = ImageEmbedder(
            "models/vit-base-patch16-224", frozen=False, hs_mean_dim=1
        )
        self.preprocessor = embedder.preprocessor
        hidden_size = 128 if is_old else 512
        self.model = FrameRNN(
            embedder, 2, hidden_size, len(dataset.label2id.keys()), is_old=is_old
        )
