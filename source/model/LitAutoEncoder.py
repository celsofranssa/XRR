import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
