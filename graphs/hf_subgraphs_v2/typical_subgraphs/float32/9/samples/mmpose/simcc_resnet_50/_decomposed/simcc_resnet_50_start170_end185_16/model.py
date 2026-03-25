import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, in_0, in_1):
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
        tmp_16 = w_16
        tmp_17 = w_17
        tmp_18 = w_18
        tmp_19 = w_19
        tmp_20 = w_20
        in_1 += in_0
        tmp_21 = in_1
        tmp_22 = torch.nn.functional.relu(tmp_21, inplace=True)
        tmp_21 = None
        tmp_23 = torch.conv_transpose2d(tmp_22, tmp_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_22 = tmp_0 = None
        tmp_24 = torch.nn.functional.batch_norm(tmp_23, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_23 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_25 = torch.nn.functional.relu(tmp_24, inplace=True)
        tmp_24 = None
        tmp_26 = torch.conv_transpose2d(tmp_25, tmp_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_25 = tmp_5 = None
        tmp_27 = torch.nn.functional.batch_norm(tmp_26, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 1e-05)
        tmp_26 = tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        tmp_28 = torch.nn.functional.relu(tmp_27, inplace=True)
        tmp_27 = None
        tmp_29 = torch.conv_transpose2d(tmp_28, tmp_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_28 = tmp_10 = None
        tmp_30 = torch.nn.functional.batch_norm(tmp_29, tmp_11, tmp_12, tmp_14, tmp_13, False, 0.1, 1e-05)
        tmp_29 = tmp_11 = tmp_12 = tmp_14 = tmp_13 = None
        tmp_31 = torch.nn.functional.relu(tmp_30, inplace=True)
        tmp_30 = None
        tmp_32 = torch.conv2d(tmp_31, tmp_16, tmp_15, (1, 1), (0, 0), (1, 1), 1)
        tmp_31 = tmp_16 = tmp_15 = None
        tmp_33 = torch.flatten(tmp_32, 2)
        tmp_32 = None
        tmp_34 = torch.nn.functional.linear(tmp_33, tmp_18, tmp_17)
        tmp_18 = tmp_17 = None
        tmp_35 = torch.nn.functional.linear(tmp_33, tmp_20, tmp_19)
        tmp_33 = tmp_20 = tmp_19 = None
        return (tmp_34, tmp_35)