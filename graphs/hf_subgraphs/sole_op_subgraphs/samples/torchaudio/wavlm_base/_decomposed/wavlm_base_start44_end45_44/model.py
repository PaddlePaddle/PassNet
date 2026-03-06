import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0 / 2.302585092994046
        return (tmp_0,)