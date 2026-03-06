import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(240, 1, 2, 2)
        return (tmp_0,)