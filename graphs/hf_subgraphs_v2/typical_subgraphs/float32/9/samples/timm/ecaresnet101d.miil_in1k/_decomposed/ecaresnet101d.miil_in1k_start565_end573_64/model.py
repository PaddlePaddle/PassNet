import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = in_1.mean((2, 3))
        tmp_2 = tmp_1.view(1, 1, -1)
        tmp_1 = None
        tmp_3 = torch.conv1d(tmp_2, tmp_0, None, (1,), (3,), (1,), 1)
        tmp_2 = tmp_0 = None
        tmp_4 = tmp_3.sigmoid()
        tmp_3 = None
        tmp_5 = tmp_4.view(1, -1, 1, 1)
        tmp_4 = None
        tmp_6 = tmp_5.expand_as(in_1)
        tmp_5 = None
        tmp_7 = in_1 * tmp_6
        tmp_6 = None
        tmp_7 += in_0
        tmp_8 = tmp_7
        tmp_7 = None
        return (tmp_8,)