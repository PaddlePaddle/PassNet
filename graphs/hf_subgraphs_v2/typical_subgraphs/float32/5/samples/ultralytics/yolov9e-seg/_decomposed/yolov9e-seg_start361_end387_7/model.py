import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14):
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
        tmp_10 = torch.nn.functional.silu(in_10, inplace=True)
        tmp_11 = torch.conv2d(in_11, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_12 = tmp_11.split([64], dim=1)
        tmp_11 = None
        tmp_13 = tmp_12[0]
        tmp_12 = None
        tmp_14 = torch.conv2d(in_12, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_3 = tmp_2 = None
        tmp_15 = tmp_14.split([64, 128], dim=1)
        tmp_14 = None
        tmp_16 = tmp_15[0]
        tmp_17 = tmp_15[1]
        tmp_15 = None
        tmp_18 = torch.conv2d(in_13, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_5 = tmp_4 = None
        tmp_19 = tmp_18.split([64, 128, 256], dim=1)
        tmp_18 = None
        tmp_20 = tmp_19[0]
        tmp_21 = tmp_19[1]
        tmp_22 = tmp_19[2]
        tmp_19 = None
        tmp_23 = torch.conv2d(in_14, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_24 = tmp_23.split([64, 128, 256, 512], dim=1)
        tmp_23 = None
        tmp_25 = tmp_24[0]
        tmp_26 = tmp_24[1]
        tmp_27 = tmp_24[2]
        tmp_28 = tmp_24[3]
        tmp_24 = None
        tmp_29 = torch.conv2d(tmp_10, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_10 = tmp_9 = tmp_8 = None
        tmp_30 = tmp_29.split([64, 128, 256, 512, 1024], dim=1)
        tmp_29 = None
        tmp_31 = tmp_30[0]
        tmp_32 = tmp_30[1]
        tmp_33 = tmp_30[2]
        tmp_34 = tmp_30[3]
        tmp_35 = tmp_30[4]
        tmp_30 = None
        return (tmp_13, tmp_16, tmp_17, tmp_20, tmp_21, tmp_22, tmp_25, tmp_26, tmp_27, tmp_28, tmp_31, tmp_32, tmp_33, tmp_34, tmp_35)