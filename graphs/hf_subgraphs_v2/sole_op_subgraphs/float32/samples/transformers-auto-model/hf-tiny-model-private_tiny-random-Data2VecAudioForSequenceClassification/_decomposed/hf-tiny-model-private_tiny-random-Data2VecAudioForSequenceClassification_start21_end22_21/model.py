import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.conv1d(in_0, tmp_1, tmp_0, (1,), (9,), (1,), 2)
        tmp_1 = tmp_0 = None
        return (tmp_2,)