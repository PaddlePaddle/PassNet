import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = tmp_1.__eq__(32)
        tmp_1 = None
        tmp_3 = tmp_2.unsqueeze(-1)
        tmp_2 = None
        tmp_4 = in_2.masked_fill(tmp_3, 0.0)
        tmp_3 = None
        tmp_5 = tmp_0.sum(-1)
        tmp_0 = None
        return (tmp_4, tmp_5)