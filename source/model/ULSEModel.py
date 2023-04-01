import torch
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import MetricCollection, F1Score


class ULSEModel(LightningModule):

    def __init__(self, hparams):

        super(ULSEModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.encoder = instantiate(hparams.encoder)

        # classification head
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(hparams.rpr_dim, hparams.vocab_size),
            torch.nn.LogSoftmax(dim=-1)
        )

        # metrics
        self.val_metrics = self._get_metrics(prefix="val_")

        # loss function
        self.loss = torch.nn.NLLLoss()

    def _get_metrics(self, prefix):
        return MetricCollection(
            metrics={
                "Mic-F1": F1Score(task="binary", num_classes=self.hparams.vocab_size, average="micro"),
                "Mac-F1": F1Score(task="binary", num_classes=self.hparams.vocab_size, average="macro"),
            },
            prefix=prefix)

    def forward(self, text, labels):
        text_rpr = self.encoder(text)
        labels_rpr = self.encoder(labels)
        rpr = 5 * text_rpr + .5 * labels_rpr
        return self.cls_head(rpr)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        text, labels, true_cls = batch["text"], batch["labels"], batch["cls"]
        pred_cls = self(text, labels)
        # log training loss
        train_loss = self.loss(pred_cls, true_cls)
        self.log('train_loss', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        text, labels, true_cls = batch["text"], batch["labels"], batch["cls"]
        pred_cls = self(text, labels)

        # log val metrics
        self.log_dict(self.val_metrics(torch.argmax(pred_cls, dim=-1), true_cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.val_metrics.compute()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx == 0:
            return {
                "text_idx": batch["text_idx"],
                "text_rpr": self.encoder(batch["text"]),
                "modality": "text"
            }
        else:
            return {
                "label_idx": batch["label_idx"],
                "label_rpr": self.encoder(batch["label"]),
                "modality": "label"
            }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)
        return optimizer

