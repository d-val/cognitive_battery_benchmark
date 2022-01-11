import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


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


class ImageDataloader(BenchmarkDataset):
    def __init__(
        self,
        folder,
        train_split=0.7,
        batch_size=128,
        label_mapping={"left": 0, "right": 2, "equal": 1},
        size=[299, 299],
        seed=42,
    ):
        data_transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        ds = BenchmarkDataset(
            folder,
            "RelativeNumbers",
            transform=lambda x: data_transforms(x.float()).to(device),
            target_transform=lambda x: torch.Tensor([label_mapping[x]])
            .long()
            .to(device),
        )
        sp = int(ds.len * (1 - train_split))
        ds_split = [ds.len - sp, sp]
        training_data, test_data = torch.utils.data.random_split(
            ds, ds_split, generator=torch.Generator().manual_seed(seed)
        )

        self.train_dataloader = DataLoader(
            training_data, batch_size=batch_size, shuffle=True
        )
        self.test_dataloader = DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )
