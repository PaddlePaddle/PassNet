import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22):
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
        tmp_17 = in_17
        tmp_18 = in_18
        tmp_19 = in_19
        tmp_20 = in_20
        tmp_21 = torch.nn.functional.adaptive_avg_pool2d(in_22, 1)
        tmp_22 = torch.conv2d(tmp_21, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_21 = tmp_1 = tmp_0 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.conv2d(tmp_23, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_23 = tmp_3 = tmp_2 = None
        tmp_25 = torch.sigmoid(tmp_24)
        tmp_24 = None
        tmp_26 = in_22 * tmp_25
        tmp_25 = None
        tmp_26 += in_21
        tmp_27 = tmp_26
        tmp_26 = None
        tmp_28 = torch.nn.functional.relu(tmp_27, inplace=True)
        tmp_27 = None
        tmp_29 = torch.conv_transpose2d(tmp_28, tmp_4, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_28 = tmp_4 = None
        tmp_30 = torch.nn.functional.batch_norm(tmp_29, tmp_5, tmp_6, tmp_8, tmp_7, False, 0.1, 1e-05)
        tmp_29 = tmp_5 = tmp_6 = tmp_8 = tmp_7 = None
        tmp_31 = torch.nn.functional.relu(tmp_30, inplace=True)
        tmp_30 = None
        tmp_32 = torch.conv_transpose2d(tmp_31, tmp_9, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_31 = tmp_9 = None
        tmp_33 = torch.nn.functional.batch_norm(tmp_32, tmp_10, tmp_11, tmp_13, tmp_12, False, 0.1, 1e-05)
        tmp_32 = tmp_10 = tmp_11 = tmp_13 = tmp_12 = None
        tmp_34 = torch.nn.functional.relu(tmp_33, inplace=True)
        tmp_33 = None
        tmp_35 = torch.conv_transpose2d(tmp_34, tmp_14, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_34 = tmp_14 = None
        tmp_36 = torch.nn.functional.batch_norm(tmp_35, tmp_15, tmp_16, tmp_18, tmp_17, False, 0.1, 1e-05)
        tmp_35 = tmp_15 = tmp_16 = tmp_18 = tmp_17 = None
        tmp_37 = torch.nn.functional.relu(tmp_36, inplace=True)
        tmp_36 = None
        tmp_38 = torch.conv2d(tmp_37, tmp_20, tmp_19, (1, 1), (0, 0), (1, 1), 1)
        tmp_37 = tmp_20 = tmp_19 = None
        return (tmp_38,)