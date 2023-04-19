import torch
from pytorch_lightning import LightningModule


class MutualAttention(LightningModule):

    def __init__(self):
        super(MutualAttention, self).__init__()
        self.r_softmax = torch.nn.Softmax(dim=-1)
        self.c_softmax = torch.nn.Softmax(dim=-2)

    def compute_mat(self, text_rpr, label_rpr):
        m = torch.einsum('b i j, c k j -> b c i k', text_rpr, label_rpr)
        m = torch.max(m, -1).values.sum(dim=-1)
        return torch.nn.functional.normalize(m, p=2, dim=-1)

    def forward(self, text_rpr, label_rpr):
        m = torch.einsum('bi,bj->bij', text_rpr, label_rpr) #self.compute_mat(text_rpr, label_rpr)
        c = self.c_softmax(m)
        r = self.r_softmax(m)

        # print(f"c({c.shape}):\n{c}\n")
        # print(f"r({r.shape}):\n{r}\n")

        a, b = torch.mean(c, -1), torch.mean(r, -2)

        return a, b
