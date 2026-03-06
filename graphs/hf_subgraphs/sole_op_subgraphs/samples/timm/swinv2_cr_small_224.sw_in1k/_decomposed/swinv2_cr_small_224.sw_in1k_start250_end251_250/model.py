import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 14, 2, 14, 2, 192)
        return (tmp_0,)