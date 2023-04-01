import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.dataset.LabelDataset import LabelDataset
from source.dataset.TextDataset import TextDataset
from source.dataset.ULSEDataset import ULSEDataset


class ULSEDataModule(pl.LightningDataModule):

    def __init__(self, params, tokenizer, vocabulary, idf, fold_idx):
        super(ULSEDataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.idf = idf
        self.fold_idx = fold_idx

    def prepare_data(self):
        with open(f"{self.params.dir}samples.pkl", "rb") as samples_file:
            self.samples = pickle.load(samples_file)

    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = ULSEDataset(
                samples=self.samples,
                ids_paths=[self.params.dir + f"fold_{self.fold_idx}/train.pkl"],
                tokenizer=self.tokenizer,
                vocabulary=self.vocabulary,
                idf=self.idf,
                window_size=self.params.window_size,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

            self.val_dataset = ULSEDataset(
                samples=self.samples,
                ids_paths=[self.params.dir + f"fold_{self.fold_idx}/val.pkl"],
                tokenizer=self.tokenizer,
                vocabulary=self.vocabulary,
                idf=self.idf,
                window_size=self.params.window_size,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

        if stage == 'test' or stage == "predict":
            self.text_dataset = TextDataset(
                samples=self.samples,
                ids_paths=[self.params.dir + f"fold_{self.fold_idx}/test.pkl"],
                tokenizer=self.tokenizer,
                vocabulary=self.vocabulary,
                text_max_length=self.params.text_max_length
            )
            self.label_dataset = LabelDataset(
                samples=self.samples,
                ids_paths=[self.params.dir + f"fold_{self.fold_idx}/test.pkl"],
                tokenizer=self.tokenizer,
                vocabulary=self.vocabulary,
                label_max_lenght=self.params.label_max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def predict_dataloader(self):
        return [
            DataLoader(self.text_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers),
            DataLoader(self.label_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers),
        ]
