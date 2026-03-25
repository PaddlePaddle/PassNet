import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1):
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
        tmp_12 = torch.prelu(in_1, tmp_5)
        tmp_5 = None
        tmp_13 = torch.conv2d(tmp_12, tmp_10, None, (1, 1), (1, 1), (1, 1), 32)
        tmp_10 = None
        tmp_14 = torch.conv2d(tmp_12, tmp_11, None, (1, 1), (2, 2), (2, 2), 32)
        tmp_12 = tmp_11 = None
        tmp_15 = torch.cat([tmp_13, tmp_14], 1)
        tmp_13 = tmp_14 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
        tmp_15 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_17 = torch.prelu(tmp_16, tmp_0)
        tmp_16 = tmp_0 = None
        tmp_18 = torch.nn.functional.adaptive_avg_pool2d(tmp_17, 1)
        tmp_19 = tmp_18.view(1, 64)
        tmp_18 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_7, tmp_6)
        tmp_19 = tmp_7 = tmp_6 = None
        tmp_21 = torch.nn.functional.relu(tmp_20, inplace=True)
        tmp_20 = None
        tmp_22 = torch.nn.functional.linear(tmp_21, tmp_9, tmp_8)
        tmp_21 = tmp_9 = tmp_8 = None
        tmp_23 = torch.sigmoid(tmp_22)
        tmp_22 = None
        tmp_24 = tmp_23.view(1, 64, 1, 1)
        tmp_23 = None
        tmp_25 = tmp_17 * tmp_24
        tmp_17 = tmp_24 = None
        tmp_26 = in_0 + tmp_25
        tmp_25 = None
        return (tmp_26,)