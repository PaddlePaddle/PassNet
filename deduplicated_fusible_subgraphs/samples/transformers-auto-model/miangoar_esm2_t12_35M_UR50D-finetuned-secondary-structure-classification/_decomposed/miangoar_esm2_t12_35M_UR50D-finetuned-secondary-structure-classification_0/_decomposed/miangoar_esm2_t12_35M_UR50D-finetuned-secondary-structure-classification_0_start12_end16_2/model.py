import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_1.__eq__(32)
        tmp_1 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_2 = in_2.masked_fill(tmp_1, 0.0)
        tmp_1 = None
        tmp_3 = in_0.sum(-1)
        return (tmp_2, tmp_3)