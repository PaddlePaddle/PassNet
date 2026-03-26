import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, in_0):
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
        tmp_12 = w_12
        tmp_13 = torch.nn.functional.silu(in_0, inplace=True)
        tmp_14 = torch.functional.split(tmp_13, [144, 144, 144, 144, 144], 1)
        tmp_13 = None
        tmp_15 = tmp_14[0]
        tmp_16 = tmp_14[1]
        tmp_17 = tmp_14[2]
        tmp_18 = tmp_14[3]
        tmp_19 = tmp_14[4]
        tmp_14 = None
        tmp_20 = torch.conv2d(tmp_15, tmp_4, None, (2, 2), (1, 1), (1, 1), 144)
        tmp_15 = tmp_4 = None
        tmp_21 = torch.conv2d(tmp_16, tmp_5, None, (2, 2), (2, 2), (1, 1), 144)
        tmp_16 = tmp_5 = None
        tmp_22 = torch.conv2d(tmp_17, tmp_6, None, (2, 2), (3, 3), (1, 1), 144)
        tmp_17 = tmp_6 = None
        tmp_23 = torch.conv2d(tmp_18, tmp_7, None, (2, 2), (4, 4), (1, 1), 144)
        tmp_18 = tmp_7 = None
        tmp_24 = torch.conv2d(tmp_19, tmp_8, None, (2, 2), (5, 5), (1, 1), 144)
        tmp_19 = tmp_8 = None
        tmp_25 = torch.cat([tmp_20, tmp_21, tmp_22, tmp_23, tmp_24], 1)
        tmp_20 = tmp_21 = tmp_22 = tmp_23 = tmp_24 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_25 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_27 = torch.nn.functional.silu(tmp_26, inplace=True)
        tmp_26 = None
        tmp_28 = tmp_27.mean((2, 3), keepdim=True)
        tmp_29 = torch.conv2d(tmp_28, tmp_12, tmp_11, (1, 1), (0, 0), (1, 1), 1)
        tmp_28 = tmp_12 = tmp_11 = None
        tmp_30 = torch.nn.functional.silu(tmp_29, inplace=True)
        tmp_29 = None
        tmp_31 = torch.conv2d(tmp_30, tmp_10, tmp_9, (1, 1), (0, 0), (1, 1), 1)
        tmp_30 = tmp_10 = tmp_9 = None
        tmp_32 = torch.sigmoid(tmp_31)
        tmp_31 = None
        tmp_33 = tmp_27 * tmp_32
        tmp_27 = tmp_32 = None
        return (tmp_33,)