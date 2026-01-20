import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0.masked_fill(in_1, 0.0)
        tmp_1 = tmp_0.view(1, 625, 8, 32)
        tmp_0 = None
        return (tmp_1,)