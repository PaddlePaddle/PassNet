import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(-1, 1, 1, 7, 7, 3072)
        return (tmp_0,)