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
        tmp_11 = torch.nn.functional.relu(in_13, inplace=True)
        tmp_12 = torch.conv2d(tmp_11, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_13 = torch.conv2d(tmp_11, tmp_1, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_1 = None
        tmp_14 = in_12 + tmp_12
        tmp_12 = None
        tmp_15 = torch.cat([in_11, tmp_13], dim=1)
        tmp_13 = None
        tmp_16 = torch.cat((tmp_14, tmp_15), dim=1)
        tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 0.001)
        tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=True)
        tmp_17 = None
        tmp_19 = torch.conv2d(tmp_18, tmp_10, None, (2, 2), (0, 0), (1, 1), 1)
        tmp_18 = tmp_10 = None
        tmp_20 = tmp_19[slice(None, None, None), slice(None, 256, None), slice(None, None, None), slice(None, None, None)]
        tmp_21 = tmp_19[slice(None, None, None), slice(256, None, None), slice(None, None, None), slice(None, None, None)]
        tmp_19 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_16, tmp_2, tmp_3, tmp_5, tmp_4, False, 0.1, 0.001)
        tmp_16 = tmp_2 = tmp_3 = tmp_5 = tmp_4 = None
        return (tmp_22, tmp_20, tmp_21)