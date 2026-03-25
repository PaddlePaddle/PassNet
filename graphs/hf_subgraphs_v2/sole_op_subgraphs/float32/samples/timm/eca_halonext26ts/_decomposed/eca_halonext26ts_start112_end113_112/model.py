import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.unfold(3, 12, 8)
        return (tmp_0,)