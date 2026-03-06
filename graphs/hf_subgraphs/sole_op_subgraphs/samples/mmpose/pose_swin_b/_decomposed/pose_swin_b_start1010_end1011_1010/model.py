import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(6, 49, 3, 16, 32)
        return (tmp_0,)