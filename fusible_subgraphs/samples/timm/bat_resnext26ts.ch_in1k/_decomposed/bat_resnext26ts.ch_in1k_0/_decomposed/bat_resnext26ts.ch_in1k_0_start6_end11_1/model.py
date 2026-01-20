import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0.view(1, 2, 8, 8)
        tmp_0 = None
        tmp_2 = tmp_1.sigmoid()
        tmp_1 = None
        tmp_3 = in_1.sum(dim=3, keepdim=True)
        tmp_4 = in_1 / tmp_3
        tmp_3 = None
        return (tmp_2, tmp_4)