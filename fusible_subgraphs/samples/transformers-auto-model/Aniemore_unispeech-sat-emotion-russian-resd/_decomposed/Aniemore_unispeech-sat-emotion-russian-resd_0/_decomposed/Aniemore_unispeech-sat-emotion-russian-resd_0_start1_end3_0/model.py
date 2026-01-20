import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = torch.conv1d(in_0, w_0, None, (5,), (0,), (1,), 1)
        tmp_1 = tmp_0.transpose(-2, -1)
        tmp_0 = None
        return (tmp_1,)