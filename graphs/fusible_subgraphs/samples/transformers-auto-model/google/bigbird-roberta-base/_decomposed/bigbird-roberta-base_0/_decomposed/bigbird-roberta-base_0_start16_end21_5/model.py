import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0 - in_1
        tmp_1 = tmp_0.split(1, dim=-1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_1 = None
        tmp_4 = tmp_2.squeeze(-1)
        tmp_2 = None
        return (tmp_3, tmp_4)