import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0.view((16, 4, -1, 16))
        return (tmp_0,)