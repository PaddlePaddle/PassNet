import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(96, 197, 3, 12, 64)
        return (tmp_0,)