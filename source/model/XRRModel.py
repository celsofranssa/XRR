import torch
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import MetricCollection, F1Score

from source.loss.SimCSE import SimCSE


class XRRModel(LightningModule):
    """Encodes the text and label into an same space of embeddings."""

    def __init__(self, hparams):

        super(XRRModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.encoder = instantiate(hparams.encoder)

        # mutual attention network
        self.man = instantiate(hparams.man)

        # embedding layer
        self.embeddings = torch.nn.Embedding(
            self.hparams.num_labels,
            self.hparams.embedding_dim)

        # dropout layers
        self.dropout = torch.nn.Dropout(hparams.dropout)

        # rpr heads
        self.rpr_head_r = torch.nn.Sequential(
            torch.nn.Dropout(hparams.dropout_1),
            torch.nn.Linear(hparams.hidden_size, hparams.hidden_size)
        )

        self.rpr_head_l = torch.nn.Sequential(
            torch.nn.Dropout(hparams.dropout_2),
            torch.nn.Linear(hparams.hidden_size, hparams.hidden_size)
        )

        self.label_rpr_head = torch.nn.Sequential(
            torch.nn.Linear(hparams.hidden_size + hparams.embedding_dim, hparams.hidden_size)
        )

        # classification head
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(4 * hparams.hidden_size, hparams.num_classes)
        )

        # metrics
        self.train_metrics = self._get_metrics(prefix="train_")
        self.val_metrics = self._get_metrics(prefix="val_")

        # loss function
        self.loss_1 = SimCSE()
        self.loss_2 = torch.nn.CrossEntropyLoss()

    def _get_metrics(self, prefix):
        return MetricCollection(
            metrics={

                "Mic-F1": F1Score(task="binary", num_classes=self.hparams.num_classes, average="micro"),
                "Mac-F1": F1Score(task="binary", num_classes=self.hparams.num_classes, average="macro"),
            },
            prefix=prefix)

    def forward(self, text, label_idx, label):

        # text rpr
        text_rpr = self.encoder(text)

        # label rpr
        label_rpr = self.encoder(label)
        label_idx_rpr = self.embeddings(label_idx)
        label_rpr = self.label_rpr_head(
            torch.cat([label_rpr, label_idx_rpr], dim=-1)
        )

        # print(f"text_rpr ({text_rpr.shape}):\n{text_rpr}\n")
        # print(f"label_rpr ({label_rpr.shape}):\n{label_rpr}\n")

        # man
        text_att, label_att = self.man(text_rpr, label_rpr)

        # print(f"text_att ({text_att.shape}):\n{text_att}\n")
        # print(f"label_att ({label_att.shape}):\n{label_att}\n")

        # cls
        rpr = torch.concat([text_rpr, text_att, label_att, label_rpr], dim=-1)
        return self.cls_head(
            self.dropout(rpr)
        )

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        text, label_idx, label, true_cls = batch["text"], batch["label_idx"], batch["label"], batch["cls"]

        # text rpr
        text_rpr = self.encoder(text)
        text_rpr_r = self.rpr_head_r(text_rpr)
        text_rpr_l = self.rpr_head_l(text_rpr)

        # label rpr
        label_rpr = self.encoder(label)
        label_idx_rpr = self.embeddings(label_idx)
        label_rpr = self.label_rpr_head(
            torch.cat([label_rpr, label_idx_rpr], dim=-1)
        )

        # man
        text_att, label_att = self.man(text_rpr, label_rpr)

        # cls
        rpr = torch.concat([text_rpr_r, text_att, label_att, label_rpr], dim=-1)
        pred_cls = self.cls_head(
            self.dropout(rpr)
        )
        # print(f"true_cls({true_cls.shape}):\n{true_cls}\n")
        # print(f"pred_cls({pred_cls.shape}):\n{pred_cls}\n")

        # log training loss
        train_loss = self.loss_1(text_rpr_r, text_rpr_l) + self.loss_2(pred_cls, true_cls)
        self.log('train_loss', train_loss)

        return train_loss

    # def training_step(self, batch, batch_idx, optimizer_idx=None):
    #     text, label, true_cls = batch["text"], batch["label"], batch["cls"]
    #     pred_cls = self(text, label)
    #     # log training loss
    #     train_loss = self.loss(pred_cls, true_cls)
    #     self.log('train_loss', train_loss)
    #
    #     return train_loss

    def validation_step(self, batch, batch_idx):
        text, label_idx, label, true_cls = batch["text"], batch["label_idx"], batch["label"], batch["cls"]
        pred_cls = self(text, label_idx, label)

        # log val metrics
        self.log_dict(self.val_metrics(torch.argmax(pred_cls, dim=-1), true_cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.val_metrics.compute()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pred_cls = self(batch["text"], batch["label_idx"], batch["label"])
        # print(f"pred_cls({pred_cls.shape}):\n{pred_cls}\n")
        return {
            "text_idx": batch["text_idx"],
            "label_idx": batch["label_idx"],
            "score": torch.nn.functional.softmax(pred_cls, dim=-1)[:, -1]
        }

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
    #                              eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)
        return optimizer
        # # schedulers
        # step_size_up = round(0.3 * self.trainer.estimated_stepping_batches)
        #
        # # scheduler = get_scheduler(
        # #     "linear",
        # #     optimizer=optimizer,
        # #     num_warmup_steps=0,
        # #     num_training_steps=self.trainer.estimated_stepping_batches
        # # )
        #
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2',
        #                                               base_lr=self.hparams.base_lr,
        #                                               max_lr=self.hparams.max_lr, step_size_up=step_size_up,
        #                                               cycle_momentum=False)
        #
        # return (
        #     {"optimizer": optimizer,
        #      "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}},
        # )
