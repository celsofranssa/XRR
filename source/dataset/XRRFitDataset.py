import pickle
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class XRRFitDataset(Dataset):
    def __init__(self, samples, ids_paths, tokenizer, text_max_length, label_max_length):
        super(XRRFitDataset, self).__init__()
        self.samples = []
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.label_max_length = label_max_length
        self._load_ids(ids_paths)

        texts = {}
        labels = {}
        labels_ids = []

        for sample_idx in tqdm(self.ids, desc="Reading samples"):
            texts[samples[sample_idx]["text_idx"]] = samples[sample_idx]["text"]
            for label_idx, label in zip(samples[sample_idx]["labels_ids"], samples[sample_idx]["labels"]):
                labels[label_idx] = label
                labels_ids.append(label_idx)

        for sample_idx in tqdm(self.ids, desc="Reshaping data"):
            for label_idx, label in zip(samples[sample_idx]["labels_ids"], samples[sample_idx]["labels"]):
                pos_sample = {
                    "text_idx": samples[sample_idx]["text_idx"],
                    "text": samples[sample_idx]["text"],
                    "label_idx": label_idx,
                    "label": label,
                    "cls": 1
                }

                self.samples.append(pos_sample)

                neg_label_idx = random.choice(labels_ids)
                while neg_label_idx in samples[sample_idx]["labels_ids"]:
                    neg_label_idx = random.choice(labels_ids)
                neg_sample = {
                    "text_idx": samples[sample_idx]["text_idx"],
                    "text": samples[sample_idx]["text"],
                    "label_idx": neg_label_idx,
                    "label": labels[neg_label_idx],
                    "cls": 0
                }
                # print(len(neg_sample["label"].split()))
                self.samples.append(neg_sample)

    def _load_ids(self, ids_paths):
        self.ids = []
        for path in ids_paths:
            with open(path, "rb") as ids_file:
                self.ids.extend(pickle.load(ids_file))

    def _encode(self, sample):
        return {
            "text_idx": sample["text_idx"],
            "text": torch.tensor(
                self.tokenizer.encode(
                    text=sample["text"], max_length=self.text_max_length, padding="max_length", truncation=True
                )),
            "label_idx": sample["label_idx"],
            "label": torch.tensor(
                self.tokenizer.encode(
                    text=sample["label"], max_length=self.label_max_length, padding="max_length", truncation=True
                )),
            "cls": sample["cls"]
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )
