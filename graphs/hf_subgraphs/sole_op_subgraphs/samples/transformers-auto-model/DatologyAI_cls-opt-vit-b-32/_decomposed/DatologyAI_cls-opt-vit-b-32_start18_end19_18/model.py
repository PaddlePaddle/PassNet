import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.__eq__(-3.4028234663852886e+38)
        return (tmp_0,)