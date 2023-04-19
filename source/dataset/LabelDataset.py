import pickle
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LabelDataset(Dataset):
    """Label Dataset.
    """

    def __init__(self, samples, ids_paths, tokenizer, vocabulary, padding_idx, label_max_length):
        super(LabelDataset, self).__init__()
        self.labels = []
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.padding_idx = padding_idx
        self.label_max_length = label_max_length
        self._load_ids(ids_paths)

        labels_map = {}
        for sample_idx in tqdm(self.ids, desc="Reading labels"):
            for label_idx, label in zip(samples[sample_idx]["labels_ids"], samples[sample_idx]["labels"]):
                labels_map[label_idx] = label

        for label_idx, label in labels_map.items():
            self.labels.append({
                "label_idx": label_idx,
                "label": label
            })

    def _load_ids(self, ids_paths):
        self.ids = []
        for path in ids_paths:
            with open(path, "rb") as ids_file:
                self.ids.extend(pickle.load(ids_file))

    def _get_tokens_ids(self, context, max_length, padding_idx):
        tokens_ids = []
        for token in self.tokenizer(context):
            tokens_ids.append(self.vocabulary.get(token.lower(), padding_idx))
        return tokens_ids[:max_length] + [padding_idx] * (max_length - len(tokens_ids))

    def _encode(self, sample):
        return {
            "label_idx": sample["label_idx"],
            "label": torch.tensor(self._get_tokens_ids(sample["label"], self.label_max_length, self.padding_idx))
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self._encode(
            self.labels[idx]
        )
