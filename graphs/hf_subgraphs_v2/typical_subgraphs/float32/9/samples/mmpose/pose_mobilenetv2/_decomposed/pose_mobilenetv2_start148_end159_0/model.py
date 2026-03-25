import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, in_0):
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
        tmp_17 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
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