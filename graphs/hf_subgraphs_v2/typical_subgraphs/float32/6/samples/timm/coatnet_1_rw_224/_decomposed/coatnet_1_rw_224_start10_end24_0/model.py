import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15):
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
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = in_12
        tmp_13 = in_13
        tmp_14 = torch.nn.functional.silu(in_15, inplace=True)
        tmp_15 = torch.conv2d(tmp_14, tmp_0, None, (2, 2), (1, 1), (1, 1), 256)
        tmp_14 = tmp_0 = None
        tmp_16 = tmp_15.mean((2, 3), keepdim=True)
        tmp_17 = torch.conv2d(tmp_16, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_16 = tmp_7 = tmp_6 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=True)
        tmp_17 = None
        tmp_19 = torch.conv2d(tmp_18, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_18 = tmp_9 = tmp_8 = None
        tmp_20 = tmp_19.sigmoid()
        tmp_19 = None
        tmp_21 = tmp_15 * tmp_20
        tmp_15 = tmp_20 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_21, tmp_2, tmp_3, tmp_5, tmp_4, False, 0.1, 1e-05)
        tmp_21 = tmp_2 = tmp_3 = tmp_5 = tmp_4 = None
        tmp_23 = torch.nn.functional.silu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.conv2d(tmp_23, tmp_1, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_23 = tmp_1 = None
        tmp_25 = tmp_24 + in_14
        tmp_24 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, tmp_10, tmp_11, tmp_13, tmp_12, False, 0.1, 1e-05)
        tmp_10 = tmp_11 = tmp_13 = tmp_12 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace=True)
        tmp_26 = None
        return (tmp_25, tmp_27)