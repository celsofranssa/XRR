import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ULSEDataset(Dataset):
    def __init__(self, samples, ids_paths, tokenizer, vocabulary, padding_idx, idf, window_size, text_max_length,
                 label_max_length):
        super(ULSEDataset, self).__init__()
        self.samples = []
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.idf = idf
        assert len(vocabulary) == padding_idx
        self.padding_idx = padding_idx
        self.text_max_length = text_max_length
        self.label_max_length = label_max_length
        self._load_ids(ids_paths)

        for sample_idx in tqdm(self.ids, desc="Reshaping data"):

            text = self.tokenizer(samples[sample_idx]["text"])
            for i, context in enumerate(range(len(text) - window_size + 1)):
                context = text[i: i + window_size]
                target = context.pop(int(len(context) / 2))
                self.samples.append({
                    "text": " ".join(context),
                    "labels": " ".join(samples[sample_idx]["labels"]),
                    "cls": target
                })

            labels = self.tokenizer(" ".join(samples[sample_idx]["labels"]))
            for i, context in enumerate(range(len(labels) - window_size + 1)):
                context = labels[i: i + window_size]
                target = context.pop(int(len(context) / 2))
                self.samples.append({
                    "text": samples[sample_idx]["text"],
                    "labels": " ".join(context),
                    "cls": target
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
            "text": torch.tensor(self._get_tokens_ids(
                sample["text"], self.text_max_length, self.padding_idx)),
            "labels": torch.tensor(self._get_tokens_ids(
                sample["labels"], self.label_max_length, self.padding_idx)),
            "cls": self.vocabulary.get(sample["cls"].lower(), self.padding_idx)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )
