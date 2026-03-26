import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.conv1d(in_1, tmp_0, None, (2,), (0,), (1,), 1)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(-2, -1)
        tmp_1 = None
        return (tmp_2,)