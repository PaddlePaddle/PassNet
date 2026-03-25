import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, in_0):
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
        tmp_13 = torch.prelu(in_0, tmp_6)
        tmp_6 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_11, None, (1, 1), (1, 1), (1, 1), 64)
        tmp_11 = None
        tmp_15 = torch.conv2d(tmp_13, tmp_12, None, (1, 1), (2, 2), (2, 2), 64)
        tmp_13 = tmp_12 = None
        tmp_16 = torch.cat([tmp_14, tmp_15], 1)
        tmp_14 = tmp_15 = None
        tmp_17 = torch.nn.functional.batch_norm(tmp_16, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
        tmp_16 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_18 = torch.prelu(tmp_17, tmp_0)
        tmp_17 = tmp_0 = None
        tmp_19 = torch.conv2d(tmp_18, tmp_5, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_18 = tmp_5 = None
        tmp_20 = torch.nn.functional.adaptive_avg_pool2d(tmp_19, 1)
        tmp_21 = tmp_20.view(1, 64)
        tmp_20 = None
        tmp_22 = torch.nn.functional.linear(tmp_21, tmp_8, tmp_7)
        tmp_21 = tmp_8 = tmp_7 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_10, tmp_9)
        tmp_23 = tmp_10 = tmp_9 = None
        tmp_25 = torch.sigmoid(tmp_24)
        tmp_24 = None
        tmp_26 = tmp_25.view(1, 64, 1, 1)
        tmp_25 = None
        tmp_27 = tmp_19 * tmp_26
        tmp_19 = tmp_26 = None
        return (tmp_27,)