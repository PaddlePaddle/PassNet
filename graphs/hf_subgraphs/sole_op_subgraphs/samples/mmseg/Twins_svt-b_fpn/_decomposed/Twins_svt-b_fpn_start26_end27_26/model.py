import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 361, 49, 3, 3, 32)
        return (tmp_0,)