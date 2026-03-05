import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 28, 2, 28, 2, 96)
        return (tmp_0,)