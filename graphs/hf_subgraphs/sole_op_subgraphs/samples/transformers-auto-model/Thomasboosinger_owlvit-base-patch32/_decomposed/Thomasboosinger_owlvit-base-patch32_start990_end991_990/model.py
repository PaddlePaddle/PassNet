import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.where(in_0, -3.4028234663852886e+38, in_1)
        return (tmp_0,)