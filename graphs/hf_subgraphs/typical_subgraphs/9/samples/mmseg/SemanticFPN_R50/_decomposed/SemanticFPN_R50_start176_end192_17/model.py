import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0, in_1, in_2, in_3, in_4):
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
        tmp_13 = w_13
        tmp_14 = w_14
        tmp_15 = w_15
        in_2 += in_1
        tmp_16 = in_2
        tmp_17 = torch.nn.functional.relu(tmp_16, inplace=True)
        tmp_16 = None
        tmp_18 = torch.conv2d(in_3, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_8 = None
        tmp_19 = torch.conv2d(in_4, tmp_11, tmp_10, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_10 = None
        tmp_20 = torch.conv2d(in_0, tmp_13, tmp_12, (1, 1), (0, 0), (1, 1), 1)
        tmp_13 = tmp_12 = None
        tmp_21 = torch.conv2d(tmp_17, tmp_15, tmp_14, (1, 1), (0, 0), (1, 1), 1)
        tmp_17 = tmp_15 = tmp_14 = None
        tmp_22 = torch.nn.functional.interpolate(tmp_21, (32, 32), None, 'nearest', None)
        tmp_23 = tmp_20 + tmp_22
        tmp_20 = tmp_22 = None
        tmp_24 = torch.nn.functional.interpolate(tmp_23, (64, 64), None, 'nearest', None)
        tmp_25 = tmp_19 + tmp_24
        tmp_19 = tmp_24 = None
        tmp_26 = torch.nn.functional.interpolate(tmp_25, (128, 128), None, 'nearest', None)
        tmp_27 = tmp_18 + tmp_26
        tmp_18 = tmp_26 = None
        tmp_28 = torch.conv2d(tmp_27, tmp_1, tmp_0, (1, 1), (1, 1), (1, 1), 1)
        tmp_27 = tmp_1 = tmp_0 = None
        tmp_29 = torch.conv2d(tmp_25, tmp_3, tmp_2, (1, 1), (1, 1), (1, 1), 1)
        tmp_25 = tmp_3 = tmp_2 = None
        tmp_30 = torch.conv2d(tmp_23, tmp_5, tmp_4, (1, 1), (1, 1), (1, 1), 1)
        tmp_23 = tmp_5 = tmp_4 = None
        tmp_31 = torch.conv2d(tmp_21, tmp_7, tmp_6, (1, 1), (1, 1), (1, 1), 1)
        tmp_21 = tmp_7 = tmp_6 = None
        return (tmp_28, tmp_29, tmp_30, tmp_31)