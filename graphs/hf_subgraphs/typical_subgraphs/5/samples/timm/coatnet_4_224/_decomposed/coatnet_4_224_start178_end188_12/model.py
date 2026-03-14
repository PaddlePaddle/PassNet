import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = torch.nn.functional.gelu(in_11)
        tmp_11 = tmp_10.mean((2, 3), keepdim=True)
        tmp_12 = torch.conv2d(tmp_11, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_7 = tmp_6 = None
        tmp_13 = torch.nn.functional.silu(tmp_12, inplace=True)
        tmp_12 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_13 = tmp_9 = tmp_8 = None
        tmp_15 = tmp_14.sigmoid()
        tmp_14 = None
        tmp_16 = tmp_10 * tmp_15
        tmp_10 = tmp_15 = None
        tmp_17 = torch.conv2d(tmp_16, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_16 = tmp_5 = tmp_4 = None
        tmp_18 = tmp_17 + in_10
        tmp_17 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        return (tmp_18, tmp_19)