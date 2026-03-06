import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.reshape(1, 5, 4, 7, 7, 156)
        return (tmp_0,)