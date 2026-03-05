import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(8, 32, 100, 100)
        return (tmp_0,)