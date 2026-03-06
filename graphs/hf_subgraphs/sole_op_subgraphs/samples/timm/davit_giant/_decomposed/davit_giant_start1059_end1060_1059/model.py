import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view(-1, 2, 2, 7, 7, 1536)
        return (tmp_0,)