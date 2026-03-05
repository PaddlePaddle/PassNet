import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.permute(3, 0, 1, 4, 2, 5)
        return (tmp_0,)