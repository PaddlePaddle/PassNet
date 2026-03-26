import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.mean(in_0, 1, True)
        return (tmp_0,)