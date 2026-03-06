import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 64, 8, 7, 8, 7)
        return (tmp_0,)