import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.functional.split(in_0, [72, 216], dim=1)
        tmp_1 = tmp_0[0]
        tmp_2 = tmp_0[1]
        tmp_0 = None
        return (tmp_1, tmp_2)