import torch
from pytorch_lightning import LightningModule


class MutualAttention(LightningModule):

    def __init__(self):
        super(MutualAttention, self).__init__()
        self.r_softmax = torch.nn.Softmax(dim=-1)
        self.c_softmax = torch.nn.Softmax(dim=-2)

    def forward(self, text_rpr, label_rpr):
        m = torch.einsum('bi,bj->bij', text_rpr, label_rpr)
        c = self.c_softmax(m)
        r = self.r_softmax(m)

        # print(f"c({c.shape}):\n{c}\n")
        # print(f"r({r.shape}):\n{r}\n")

        a, b = torch.mean(c, -1), torch.mean(r, -2)

        return a, b
