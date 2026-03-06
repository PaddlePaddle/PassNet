import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(20, 49, 3, 8, 32)
        return (tmp_0,)