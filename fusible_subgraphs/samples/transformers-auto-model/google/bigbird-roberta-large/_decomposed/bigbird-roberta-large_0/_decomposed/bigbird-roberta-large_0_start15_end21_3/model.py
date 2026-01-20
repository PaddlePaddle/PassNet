import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 * 1000000.0
        tmp_1 = in_1 - tmp_0
        tmp_0 = None
        tmp_2 = tmp_1.split(1, dim=-1)
        tmp_1 = None
        tmp_3 = tmp_2[0]
        tmp_4 = tmp_2[1]
        tmp_2 = None
        tmp_5 = tmp_3.squeeze(-1)
        tmp_3 = None
        return (tmp_4, tmp_5)