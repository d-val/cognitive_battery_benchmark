import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pickle5 as pickle
import numpy as np
import torchvision.models as models
from torch import nn

from torch.utils.data import DataLoader
import torch.nn.functional as F

from experiments.model_development.model import ImageClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

resnet18 = models.resnet18(pretrained=True)
num_ftrs = resnet18.fc.in_features
resnet18.out = nn.Linear(num_ftrs, 3)
resnet18 = resnet18.to(device)

ct = 0
for child in resnet18.children():
    ct += 1
    if ct < 9:
        for param in child.parameters():
            param.requires_grad = False

model = resnet18

classifier = ImageClassifier(
    model, loss_fn=nn.CrossEntropyLoss(), num_classes=3, freeze_layers=9
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_dataloader, test_dataloader = None, None  # TODO: load data

classifier.train(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    save_path="experiments/model_development/output_models/resnet18_9_layers_frozen.pt",
)
print("Done!")

torch.save(model.state_dict(), "output")
