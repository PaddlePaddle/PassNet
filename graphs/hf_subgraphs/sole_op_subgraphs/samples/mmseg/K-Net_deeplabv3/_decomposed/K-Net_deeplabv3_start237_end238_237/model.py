import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.permute(1, 0, 2)
        return (tmp_0,)