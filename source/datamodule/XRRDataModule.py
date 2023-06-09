import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.dataset.XRRFitDataset import XRRFitDataset
from source.dataset.XRRPredictDataset import XRRPredictDataset


class XRRDataModule(pl.LightningDataModule):

    def __init__(self, params, tokenizer, rankings, fold_idx):
        super(XRRDataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.rankings = rankings
        self.fold_idx = fold_idx

    def prepare_data(self):
        with open(f"{self.params.dir}samples.pkl", "rb") as samples_file:
            self.samples = pickle.load(samples_file)

    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = XRRFitDataset(
                samples=self.samples,
                ids_paths=[self.params.dir + f"fold_{self.fold_idx}/train.pkl"],
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

            self.val_dataset = XRRFitDataset(
                samples=self.samples,
                ids_paths=[self.params.dir + f"fold_{self.fold_idx}/val.pkl"],
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

        if stage == 'test' or stage == "predict":
            self.predict_dataset = XRRPredictDataset(
                samples=self.samples,
                rankings=self.rankings,
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
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
        return DataLoader(
            self.predict_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

