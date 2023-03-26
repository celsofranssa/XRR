import torch
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import MetricCollection, F1Score


class XRRModel(LightningModule):
    """Encodes the text and label into an same space of embeddings."""

    def __init__(self, hparams):

        super(XRRModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.encoder = instantiate(hparams.encoder)

        self.man = instantiate(hparams.man)

        # dropout layer
        self.dropout = torch.nn.Dropout(hparams.dropout)

        # classification head
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(4 * hparams.hidden_size, hparams.num_classes)
        )

        # metrics
        self.train_metrics = self._get_metrics(prefix="train_")
        self.val_metrics = self._get_metrics(prefix="val_")

        # loss function
        self.loss = torch.nn.CrossEntropyLoss()

    def _get_metrics(self, prefix):
        return MetricCollection(
            metrics={

                "Mic-F1": F1Score(task="binary", num_classes=self.hparams.num_classes, average="micro"),
                "Mac-F1": F1Score(task="binary", num_classes=self.hparams.num_classes, average="macro"),
            },
            prefix=prefix)

    def forward(self, text, label):
        text_rpr = self.encoder(text)
        label_rpr = self.encoder(label)

        text_att, label_att = self.man(text_rpr, label_rpr)
        rpr = torch.concat([text_rpr, text_att, label_att, label_rpr], dim=-1)

        return self.cls_head(
            self.dropout(rpr)
        )

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        text, label, true_cls = batch["text"], batch["label"], batch["cls"]
        pred_cls = self(text, label)
        # log training loss
        train_loss = self.loss(pred_cls, true_cls)
        self.log('train_loss', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        text, label, true_cls = batch["text"], batch["label"], batch["cls"]
        pred_cls = self(text, label)

        # log val metrics
        self.log_dict(self.val_metrics(torch.argmax(pred_cls, dim=-1), true_cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.val_metrics.compute()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx == 0:
            text_idx, text, = batch["text_idx"], batch["text"]
            text_rpr = self.pool(self.encoder(text))

            return {
                "text_idx": text_idx,
                "text_rpr": text_rpr,
                "modality": "text"
            }
        else:
            label_idx, label = batch["label_idx"], batch["label"]
            label_rpr = self.pool(self.encoder(label))

            return {
                "label_idx": label_idx,
                "label_rpr": label_rpr,
                "modality": "label"
            }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)
