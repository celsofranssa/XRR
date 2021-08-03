import json

import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """MNIST Dataset.
    """

    def __init__(self, dataset_path):
        self.samples = []
        self._init_dataset(dataset_path)

    def _init_dataset(self, dataset_path):
        with open(dataset_path, "r") as dataset_file:
            for line in dataset_file:
                sample = json.loads(line)
                self.samples.append({
                    "idx": sample["idx"],
                    "x": sample["x"],
                    "y": sample["y"]
                })

    def _encode(self, sample):
        return {
            "idx": sample["idx"],
            "x": torch.tensor(sample["x"], dtype=torch.float32),
            "y": sample["y"]
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(self.samples[idx])
