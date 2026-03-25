import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_1 = torch.functional.split(tmp_0, [112, 112, 112], 1)
        tmp_0 = None
        tmp_2 = tmp_1[0]
        tmp_3 = tmp_1[1]
        tmp_4 = tmp_1[2]
        tmp_1 = None
        return (tmp_2, tmp_3, tmp_4)