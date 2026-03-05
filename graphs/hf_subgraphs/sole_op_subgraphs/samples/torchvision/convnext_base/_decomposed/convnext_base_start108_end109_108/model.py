import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.permute(in_0, [0, 3, 1, 2])
        return (tmp_0,)