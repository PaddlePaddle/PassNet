import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(100, 49, 3, 12, 32)
        return (tmp_0,)