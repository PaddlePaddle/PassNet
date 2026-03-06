import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape((1, 72, 72, 1024))
        return (tmp_0,)