import torch
from torch import nn


class SimCSE(nn.Module):

    def __init__(self):
        super(SimCSE, self).__init__()

    def forward(self, rpr_r, rpr_l):
        scores = torch.matmul(rpr_r, rpr_l.t())
        diagonal_mean = torch.mean(torch.diag(scores))
        mean_log_row_sum_exp = torch.mean(torch.logsumexp(scores, dim=1))
        return -diagonal_mean + mean_log_row_sum_exp


