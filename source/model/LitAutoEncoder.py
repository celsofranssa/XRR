import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.hparams.encoder.width * self.hparams.encoder.height, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.hparams.encoder.output_size)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.decoder.input_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.hparams.decoder.width * self.hparams.decoder.height)
        )

        # loss
        self.loss = nn.MSELoss()


    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss(x_hat, x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        # optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True)
        return optimizer
