import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.unfold(2, 12, 8)
        return (tmp_0,)