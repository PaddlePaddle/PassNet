import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_2 = tmp_1.mean((2, 3))
        tmp_3 = tmp_2.view(1, 1, -1)
        tmp_2 = None
        tmp_4 = torch.conv1d(tmp_3, tmp_0, None, (1,), (1,), (1,), 1)
        tmp_3 = tmp_0 = None
        tmp_5 = tmp_4.sigmoid()
        tmp_4 = None
        tmp_6 = tmp_5.view(1, -1, 1, 1)
        tmp_5 = None
        tmp_7 = tmp_6.expand_as(tmp_1)
        tmp_6 = None
        tmp_8 = tmp_1 * tmp_7
        tmp_1 = tmp_7 = None
        return (tmp_8,)