import torch
from torch import nn


class EmbeddingEncoder(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, rpr_dim):
        super(EmbeddingEncoder, self).__init__()
        self.embeddings = nn.Embedding(
            vocabulary_size + 1,  # padding
            embedding_dim,
            padding_idx=vocabulary_size,
            scale_grad_by_freq=True
        )
        self.linear = nn.Linear(embedding_dim, rpr_dim)

    def forward(self, inputs):
        # print(f"\ninputs({inputs.shape}):\n{inputs}\n")
        # att_mask = torch.where(inputs >= 0, 1, 0)
        # embeddings = self.embeddings(inputs)
        # att_mask = att_mask.unsqueeze(-1).expand(embeddings.size()).float()
        # embeddings = torch.sum(embeddings * att_mask, 1)

        # hidden_states = encoder_outputs.last_hidden_state
        # attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        # sum_hidden_states = torch.sum(hidden_states * attention_mask, 1)
        #
        embeddings = torch.sum(self.embeddings(inputs), 1)
        return self.linear(embeddings)
