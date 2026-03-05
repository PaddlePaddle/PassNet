import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.expand([8, 124, 124])
        return (tmp_0,)