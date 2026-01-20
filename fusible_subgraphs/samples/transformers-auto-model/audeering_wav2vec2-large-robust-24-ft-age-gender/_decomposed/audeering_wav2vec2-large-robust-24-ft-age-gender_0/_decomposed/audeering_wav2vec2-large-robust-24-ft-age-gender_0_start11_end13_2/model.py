import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.conv1d(in_0, w_1, w_0, (2,), (0,), (1,), 1)
        tmp_1 = tmp_0.transpose(-2, -1)
        tmp_0 = None
        return (tmp_1,)