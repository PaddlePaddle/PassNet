import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
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
        tmp_14 = in_14
        tmp_15 = in_15
        tmp_16 = in_16
        in_18 += in_17
        tmp_17 = in_18
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=True)
        tmp_17 = None
        tmp_19 = torch.conv_transpose2d(tmp_18, tmp_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_18 = tmp_0 = None
        tmp_20 = torch.nn.functional.batch_norm(tmp_19, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_19 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_21 = torch.nn.functional.relu(tmp_20, inplace=True)
        tmp_20 = None
        tmp_22 = torch.conv_transpose2d(tmp_21, tmp_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_21 = tmp_5 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 1e-05)
        tmp_22 = tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        tmp_24 = torch.nn.functional.relu(tmp_23, inplace=True)
        tmp_23 = None
        tmp_25 = torch.conv_transpose2d(tmp_24, tmp_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_24 = tmp_10 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, tmp_11, tmp_12, tmp_14, tmp_13, False, 0.1, 1e-05)
        tmp_25 = tmp_11 = tmp_12 = tmp_14 = tmp_13 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace=True)
        tmp_26 = None
        tmp_28 = torch.conv2d(tmp_27, tmp_16, tmp_15, (1, 1), (0, 0), (1, 1), 1)
        tmp_27 = tmp_16 = tmp_15 = None
        return (tmp_28,)