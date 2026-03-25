import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = in_2.mean((2, 3))
        tmp_2 = torch.sym_sum([-1, in_1])
        tmp_3 = tmp_2 // 4
        tmp_4 = torch.sym_sum([1, tmp_3])
        tmp_3 = tmp_4 = None
        tmp_5 = tmp_1.view(1, 1, -1)
        tmp_1 = None
        tmp_6 = torch.conv1d(tmp_5, tmp_0, None, (1,), (2,), (1,), 1)
        tmp_5 = tmp_0 = None
        tmp_7 = tmp_6.sigmoid()
        tmp_6 = None
        tmp_8 = tmp_7.view(1, -1, 1, 1)
        tmp_7 = None
        tmp_9 = tmp_8.expand_as(in_2)
        tmp_8 = None
        tmp_10 = in_2 * tmp_9
        tmp_9 = None
        return (tmp_2, tmp_10)