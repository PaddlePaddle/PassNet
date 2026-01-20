import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0.view(1, 2, 8, 8)
        tmp_0 = None
        tmp_2 = tmp_1.sigmoid()
        tmp_1 = None
        return (tmp_2,)