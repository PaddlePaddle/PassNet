import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.permute(2, 0, 1, 3)
        return (tmp_0,)