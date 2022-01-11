import os

import numpy as np
import pickle5 as pickle
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class ImageClassifier:
    def __init__(
        self, model, loss_fn=nn.CrossEntropyLoss(), num_classes=3, freeze_layers=9
    ):
        if type(model) == str:
            model = models.__dict__[model](pretrained=True)
            num_ftrs = model.fc.in_features
            model.out = nn.Linear(num_ftrs, num_classes)

            for ct, child in enumerate(model.children()):
                if ct < 9:
                    for param in child.parameters():
                        param.requires_grad = False

        self.num_classes = num_classes

        self.model = model.to(device)

        self.loss_fn = nn.CrossEntropyLoss()

    def train(
        self,
        train_dataloader,
        test_dataloader,
        epochs=1000,
        optimizer=torch.optim.Adam,
        learning_rate=1e-3,
        save_path=None,
    ):

        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

        train_l, test_l, acc = [], [], []
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train_loss = self.train_loop(
                train_dataloader, self.model, self.loss_fn, self.optimizer
            )
            test_loss, correct = self.test_loop(
                test_dataloader, self.model, self.loss_fn
            )
            train_l.append(train_loss)
            test_l.append(test_loss)
        print("Done!")
        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)

    def train_loop(self, dataloader):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, torch.flatten(y))

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return loss

    def test_loop(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, torch.flatten(y)).item()
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


class BenchmarkDataset(Dataset):
    def __init__(self, folder, experiment, transform=None, target_transform=None):
        super(BenchmarkDataset).__init__()
        self.folder = folder
        self.experiment = experiment
        self.transform = transform
        self.target_transform = target_transform
        self.len = len(next(os.walk(os.path.join(self.folder, experiment)))[1])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with open(
            f"{self.folder}/{self.experiment}/{idx}/machine_readable/iteration_data.pickle",
            "rb",
        ) as iter_file:
            iter_data = pickle.load(iter_file)
        images, label = (
            torch.from_numpy(np.moveaxis(iter_data["images"][0], 2, 0)),
            iter_data["label"],
        )
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            label = self.target_transform(label)
        return images, label
