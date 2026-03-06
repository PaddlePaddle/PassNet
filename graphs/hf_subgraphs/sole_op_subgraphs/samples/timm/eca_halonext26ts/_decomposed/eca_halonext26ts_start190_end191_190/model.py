import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(32, 4, 1, 4, 12)
        return (tmp_0,)