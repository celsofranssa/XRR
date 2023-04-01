import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    """Text Dataset.
    """

    def __init__(self, samples, ids_paths, tokenizer, vocabulary, padding_idx, text_max_length):
        super(TextDataset, self).__init__()
        self.texts = []
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.padding_idx = padding_idx
        self.text_max_length = text_max_length
        self._load_ids(ids_paths)

        for sample_idx in tqdm(self.ids, desc="Reading texts"):
            self.texts.append({
                "text_idx": samples[sample_idx]["text_idx"],
                "text": samples[sample_idx]["text"]
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
            "text_idx": sample["text_idx"],
            "text": torch.tensor(self._get_tokens_ids(sample["text"], self.text_max_length, self.padding_idx))
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self._encode(
            self.texts[idx]
        )
