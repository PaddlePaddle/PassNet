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
        tmp_21 = torch.nn.functional.adaptive_avg_pool2d(in_1, 1)
        tmp_22 = torch.conv2d(tmp_21, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_21 = tmp_1 = tmp_0 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.conv2d(tmp_23, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_23 = tmp_3 = tmp_2 = None
        tmp_25 = torch.sigmoid(tmp_24)
        tmp_24 = None
        tmp_26 = in_1 * tmp_25
        tmp_25 = None
        tmp_26 += in_0
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