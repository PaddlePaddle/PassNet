import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = w_10
        tmp_11 = w_11
        tmp_12 = torch.nn.functional.silu(in_1, inplace=True)
        tmp_13 = torch.conv2d(tmp_12, tmp_2, None, (1, 1), (1, 1), (1, 1), 17)
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
        tmp_23 = tmp_22 + in_0
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