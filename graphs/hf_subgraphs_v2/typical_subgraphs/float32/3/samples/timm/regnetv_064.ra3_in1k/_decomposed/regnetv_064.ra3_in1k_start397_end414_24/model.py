import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13):
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
        tmp_12 = torch.nn.functional.silu(in_13, inplace=True)
        tmp_13 = torch.conv2d(tmp_12, tmp_2, None, (1, 1), (1, 1), (1, 1), 18)
        tmp_12 = tmp_2 = None
        tmp_14 = tmp_13.mean((2, 3), keepdim=True)
        tmp_15 = torch.conv2d(tmp_14, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_14 = tmp_9 = tmp_8 = None
        tmp_16 = torch.nn.functional.silu(tmp_15, inplace=True)
        tmp_15 = None
        tmp_17 = torch.conv2d(tmp_16, tmp_11, tmp_10, (1, 1), (0, 0), (1, 1), 1)
        tmp_16 = tmp_11 = tmp_10 = None
        tmp_18 = tmp_17.sigmoid()
        tmp_17 = None
        tmp_19 = tmp_13 * tmp_18
        tmp_13 = tmp_18 = None
        tmp_20 = torch.nn.functional.batch_norm(tmp_19, tmp_4, tmp_5, tmp_7, tmp_6, False, 0.1, 1e-05)
        tmp_19 = tmp_4 = tmp_5 = tmp_7 = tmp_6 = None
        tmp_21 = torch.nn.functional.silu(tmp_20, inplace=True)
        tmp_20 = None
        tmp_22 = torch.conv2d(tmp_21, tmp_3, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_21 = tmp_3 = None
        tmp_23 = tmp_22 + in_12
        tmp_22 = None
        tmp_24 = torch.nn.functional.silu(tmp_23, inplace=False)
        tmp_23 = None
        tmp_25 = torch.nn.functional.adaptive_avg_pool2d(tmp_24, 1)
        tmp_24 = None
        tmp_26 = tmp_25.flatten(1, -1)
        tmp_25 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.0, False, False)
        tmp_26 = None
        tmp_28 = torch.nn.functional.linear(tmp_27, tmp_1, tmp_0)
        tmp_27 = tmp_1 = tmp_0 = None
        return (tmp_28,)