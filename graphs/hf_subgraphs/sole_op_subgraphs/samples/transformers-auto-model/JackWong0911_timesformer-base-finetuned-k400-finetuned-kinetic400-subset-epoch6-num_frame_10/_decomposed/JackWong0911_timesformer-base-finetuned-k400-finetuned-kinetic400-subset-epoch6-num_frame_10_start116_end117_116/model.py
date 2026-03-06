import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.permute(0, 3, 1, 2, 4)
        return (tmp_0,)