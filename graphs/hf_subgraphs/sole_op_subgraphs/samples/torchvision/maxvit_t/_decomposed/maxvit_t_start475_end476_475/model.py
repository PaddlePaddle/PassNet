import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 256, 7, 2, 7, 2)
        return (tmp_0,)