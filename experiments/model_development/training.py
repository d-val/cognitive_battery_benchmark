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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, torch.flatten(y))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, torch.flatten(y)).item()
            correct += (
                (pred.argmax(1) == torch.flatten(y)).type(torch.float).sum().item()
            )

    test_loss /= num_batches
    print(size, correct)
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss, correct


epochs = 10
test_l = acc = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, correct = test_loop(test_dataloader, model, loss_fn)
    test_l.append(test_loss)
    acc.append(correct)
print("Done!")

torch.save(model.state_dict(), "output")
