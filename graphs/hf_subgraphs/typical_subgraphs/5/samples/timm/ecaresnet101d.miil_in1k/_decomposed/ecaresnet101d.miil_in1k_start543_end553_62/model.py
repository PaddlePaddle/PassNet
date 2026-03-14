import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_3.mean((2, 3))
        tmp_2 = in_1 // 32
        tmp_3 = torch.sym_sum([1, tmp_2])
        tmp_2 = tmp_3 = None
        tmp_4 = tmp_1.view(1, 1, -1)
        tmp_1 = None
        tmp_5 = torch.conv1d(tmp_4, tmp_0, None, (1,), (3,), (1,), 1)
        tmp_4 = tmp_0 = None
        tmp_6 = tmp_5.sigmoid()
        tmp_5 = None
        tmp_7 = tmp_6.view(1, -1, 1, 1)
        tmp_6 = None
        tmp_8 = tmp_7.expand_as(in_3)
        tmp_7 = None
        tmp_9 = in_3 * tmp_8
        tmp_8 = None
        tmp_10 = torch.nn.functional.avg_pool2d(in_2, 2, 2, 0, True, False, None)
        return (tmp_10, tmp_9)