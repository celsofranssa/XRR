from omegaconf import OmegaConf

from source.datamodule.MNISTDataModule import MNISTDataModule
from source.model.LitAutoEncoder import LitAutoEncoder
import pytorch_lightning as pl


class FitHelper:
    def __init__(self, params):
        self.params = params


    def perform_fit(self):

        # init model
        ae = LitAutoEncoder(self.params.model)

        # Initialize a trainer
        trainer = pl.Trainer(
            fast_dev_run=self.params.trainer.fast_dev_run,
            max_epochs=self.params.trainer.max_epochs,
            precision=self.params.trainer.precision,
            gpus=self.params.trainer.gpus,
        )

        for fold in self.params.data.folds:
            # load data
            dm = MNISTDataModule(self.params.data, fold=fold)

        # Train the ⚡ model
        print(f"Fitting with fowling params"
              f"{OmegaConf.to_yaml(self.params, resolve=True)}"
              )
        trainer.fit(
            model=ae,
            datamodule=dm
        )