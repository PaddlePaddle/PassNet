import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = torch.nn.functional.silu(in_1, inplace=True)
        tmp_6 = tmp_5.mean((2, 3), keepdim=True)
        tmp_7 = torch.conv2d(tmp_6, tmp_2, tmp_1, (1, 1), (0, 0), (1, 1), 1)
        tmp_6 = tmp_2 = tmp_1 = None
        tmp_8 = torch.nn.functional.silu(tmp_7, inplace=True)
        tmp_7 = None
        tmp_9 = torch.conv2d(tmp_8, tmp_4, tmp_3, (1, 1), (0, 0), (1, 1), 1)
        tmp_8 = tmp_4 = tmp_3 = None
        tmp_10 = tmp_9.sigmoid()
        tmp_9 = None
        tmp_11 = tmp_5 * tmp_10
        tmp_5 = tmp_10 = None
        tmp_12 = torch.conv2d(tmp_11, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_0 = None
        tmp_13 = tmp_12 + in_0
        tmp_12 = None
        tmp_14 = torch.nn.functional.avg_pool2d(tmp_13, 2, 2, 0, False, True, None)
        return (tmp_13, tmp_14)