import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = torch.nn.functional.gelu(in_1)
        tmp_7 = tmp_6.mean((2, 3), keepdim=True)
        tmp_8 = torch.conv2d(tmp_7, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = torch.nn.functional.silu(tmp_8, inplace=True)
        tmp_8 = None
        tmp_10 = torch.conv2d(tmp_9, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_5 = tmp_4 = None
        tmp_11 = tmp_10.sigmoid()
        tmp_10 = None
        tmp_12 = tmp_6 * tmp_11
        tmp_6 = tmp_11 = None
        tmp_13 = torch.conv2d(tmp_12, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_12 = tmp_1 = tmp_0 = None
        tmp_14 = tmp_13 + in_0
        tmp_13 = None
        tmp_15 = torch.nn.functional.avg_pool2d(tmp_14, 2, 2, 0, False, True, None)
        return (tmp_14, tmp_15)