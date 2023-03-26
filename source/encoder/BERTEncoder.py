from pytorch_lightning import LightningModule
from transformers import BertModel

class BERTEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, pooling):
        super(BERTEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(
            architecture)
        self.pooling = pooling

    def forward(self, input_ids):
        attention_mask = (input_ids > 0).int()
        encoder_outputs = self.encoder(input_ids, attention_mask)

        return self.pooling(
            encoder_outputs,
            attention_mask

        )
