import torch
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from transformers import get_constant_schedule_with_warmup, get_scheduler

from source.metric.MRRMetric import MRRMetric


class XRRModel(LightningModule):
    """Encodes the text and label into an same space of embeddings."""

    def __init__(self, hparams):

        super(SiEMTCModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.encoder = instantiate(hparams.encoder)

        self.att_encoder = instantiate(hparams.encoder)

        # dropout layer
        self.dropout = torch.nn.Dropout(hparams.dropout)

        # classification head
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(hparams.hidden_size, hparams.num_classes)
        )

        # loss function
        self.loss = instantiate(hparams.loss)

    def forward(self, text, label):
        text_rpr = self.encoder(text)
        label_rpr = self.encoder(label)
        text_att, label_att = self.att_encoder(text_rpr, label_rpr)
        rpr = torch.concat([text_rpr, text_att, label_att, label_rpr], dim=-1)

        return self.cls_head(
            self.dropout(rpr)
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
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
        self.log("val_MRR", self.mrr.compute(), prog_bar=True)
        self.mrr.reset()

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
        return torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)
