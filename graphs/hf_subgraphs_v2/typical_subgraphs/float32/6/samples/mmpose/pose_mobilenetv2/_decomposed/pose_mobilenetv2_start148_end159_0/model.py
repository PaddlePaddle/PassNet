import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
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
        tmp_17 = torch.nn.functional.hardtanh(in_17, 0.0, 6.0, True)
        tmp_18 = torch.conv_transpose2d(tmp_17, tmp_0, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_17 = tmp_0 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 1e-05)
        tmp_18 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_20 = torch.nn.functional.relu(tmp_19, inplace=True)
        tmp_19 = None
        tmp_21 = torch.conv_transpose2d(tmp_20, tmp_5, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_20 = tmp_5 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_21, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 1e-05)
        tmp_21 = tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.conv_transpose2d(tmp_23, tmp_10, None, (2, 2), (1, 1), (0, 0), 1, (1, 1))
        tmp_23 = tmp_10 = None
        tmp_25 = torch.nn.functional.batch_norm(tmp_24, tmp_11, tmp_12, tmp_14, tmp_13, False, 0.1, 1e-05)
        tmp_24 = tmp_11 = tmp_12 = tmp_14 = tmp_13 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace=True)
        tmp_25 = None
        tmp_27 = torch.conv2d(tmp_26, tmp_16, tmp_15, (1, 1), (0, 0), (1, 1), 1)
        tmp_26 = tmp_16 = tmp_15 = None
        return (tmp_27,)