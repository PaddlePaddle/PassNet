import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = torch.conv1d(in_0, tmp_0, None, (5,), (0,), (1,), 1)
        tmp_0 = None
        return (tmp_1,)