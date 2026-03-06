import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(1, 19, 19, 7, 7, -1)
        return (tmp_0,)