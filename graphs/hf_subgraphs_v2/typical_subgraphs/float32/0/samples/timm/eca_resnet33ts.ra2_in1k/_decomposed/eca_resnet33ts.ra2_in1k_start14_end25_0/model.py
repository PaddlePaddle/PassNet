import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.silu(in_2, inplace=True)
        tmp_2 = tmp_1.mean((2, 3))
        tmp_3 = torch.sym_sum([-1, in_1])
        tmp_4 = tmp_3 // 4
        tmp_5 = torch.sym_sum([1, tmp_4])
        tmp_4 = tmp_5 = None
        tmp_6 = tmp_2.view(1, 1, -1)
        tmp_2 = None
        tmp_7 = torch.conv1d(tmp_6, tmp_0, None, (1,), (1,), (1,), 1)
        tmp_6 = tmp_0 = None
        tmp_8 = tmp_7.sigmoid()
        tmp_7 = None
        tmp_9 = tmp_8.view(1, -1, 1, 1)
        tmp_8 = None
        tmp_10 = tmp_9.expand_as(tmp_1)
        tmp_9 = None
        tmp_11 = tmp_1 * tmp_10
        tmp_1 = tmp_10 = None
        return (tmp_3, tmp_11)