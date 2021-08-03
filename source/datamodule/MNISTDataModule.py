import pytorch_lightning as pl

from torch.utils.data import DataLoader

from source.dataset.MNISTDataset import MNISTDataset


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, params, fold):
        super(MNISTDataModule, self).__init__()
        self.params = params
        self.fold = fold

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = MNISTDataset(
                self.params.dir + f"fold_{self.fold}/train.jsonl"
            )

            self.val_dataset = MNISTDataset(
                self.params.dir + f"fold_{self.fold}/val.jsonl"
            )

        if stage == 'test' or stage is None:
            self.test_dataset = MNISTDataset(
                self.params.dir + f"fold_{self.fold}/test.jsonl"
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
            shuffle=False,
            num_workers=self.params.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers
        )
