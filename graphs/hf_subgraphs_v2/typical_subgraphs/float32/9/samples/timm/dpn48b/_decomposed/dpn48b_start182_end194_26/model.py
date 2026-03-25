import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, in_0, in_1, in_2):
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
        tmp_11 = torch.nn.functional.relu(in_2, inplace=True)
        tmp_12 = torch.conv2d(tmp_11, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_0 = None
        tmp_13 = torch.conv2d(tmp_11, tmp_1, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_1 = None
        tmp_14 = in_1 + tmp_12
        tmp_12 = None
        tmp_15 = torch.cat([in_0, tmp_13], dim=1)
        tmp_13 = None
        tmp_16 = torch.cat((tmp_14, tmp_15), dim=1)
        tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, tmp_6, tmp_7, tmp_9, tmp_8, False, 0.1, 0.001)
        tmp_6 = tmp_7 = tmp_9 = tmp_8 = None
        tmp_18 = torch.nn.functional.relu(tmp_17, inplace=True)
        tmp_17 = None
        tmp_19 = torch.conv2d(tmp_18, tmp_10, None, (2, 2), (0, 0), (1, 1), 1)
        tmp_18 = tmp_10 = None
        tmp_20 = tmp_19[slice(None, None, None), slice(None, 512, None), slice(None, None, None), slice(None, None, None)]
        tmp_21 = tmp_19[slice(None, None, None), slice(512, None, None), slice(None, None, None), slice(None, None, None)]
        tmp_19 = None
        tmp_22 = torch.nn.functional.batch_norm(tmp_16, tmp_2, tmp_3, tmp_5, tmp_4, False, 0.1, 0.001)
        tmp_16 = tmp_2 = tmp_3 = tmp_5 = tmp_4 = None
        return (tmp_22, tmp_20, tmp_21)