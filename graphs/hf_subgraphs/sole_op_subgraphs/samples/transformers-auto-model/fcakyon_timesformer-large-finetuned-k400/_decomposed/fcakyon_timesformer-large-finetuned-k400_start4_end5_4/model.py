import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = w_0
        tmp_1 = tmp_0.expand(96, -1, -1)
        tmp_0 = None
        return (tmp_1,)