import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(8, 80, 1, -1)
        return (tmp_0,)