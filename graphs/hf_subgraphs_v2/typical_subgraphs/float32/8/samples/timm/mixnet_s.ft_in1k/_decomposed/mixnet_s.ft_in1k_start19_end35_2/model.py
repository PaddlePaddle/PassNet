import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
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
        tmp_12 = torch.nn.functional.relu(in_12, inplace=True)
        tmp_13 = torch.functional.split(tmp_12, [48, 48], 1)
        tmp_12 = None
        tmp_14 = tmp_13[0]
        tmp_15 = tmp_13[1]
        tmp_13 = None
        tmp_16 = torch.conv2d(tmp_14, tmp_4, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_14 = tmp_4 = None
        tmp_17 = torch.conv2d(tmp_15, tmp_5, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_15 = tmp_5 = None
        tmp_18 = torch.cat([tmp_16, tmp_17], 1)
        tmp_16 = tmp_17 = None
        tmp_19 = torch.nn.functional.batch_norm(tmp_18, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
        tmp_18 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
        tmp_20 = torch.functional.split(tmp_19, [12, 12], 1)
        tmp_21 = tmp_20[0]
        tmp_22 = tmp_20[1]
        tmp_20 = None
        tmp_23 = torch.conv2d(tmp_21, tmp_10, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_21 = tmp_10 = None
        tmp_24 = torch.conv2d(tmp_22, tmp_11, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_22 = tmp_11 = None
        tmp_25 = torch.cat([tmp_23, tmp_24], 1)
        tmp_23 = tmp_24 = None
        tmp_26 = torch.nn.functional.batch_norm(tmp_25, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 1e-05)
        tmp_25 = tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        tmp_27 = torch.nn.functional.relu(tmp_26, inplace=True)
        tmp_26 = None
        return (tmp_19, tmp_27)