import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.split([32, 32, 128], dim=3)
        return (tmp_0,)