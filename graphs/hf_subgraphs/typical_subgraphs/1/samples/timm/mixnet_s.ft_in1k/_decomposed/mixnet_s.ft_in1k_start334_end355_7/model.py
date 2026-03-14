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
        tmp_12 = in_12
        tmp_13 = torch.nn.functional.silu(in_13, inplace=True)
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