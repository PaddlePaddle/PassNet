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
        tmp_11 = in_11
        tmp_12 = torch.prelu(in_13, tmp_5)
        tmp_5 = None
        tmp_13 = torch.conv2d(tmp_12, tmp_10, None, (1, 1), (1, 1), (1, 1), 64)
        tmp_10 = None
        tmp_14 = torch.conv2d(tmp_12, tmp_11, None, (1, 1), (4, 4), (4, 4), 64)
        tmp_12 = tmp_11 = None
        tmp_15 = torch.cat([tmp_13, tmp_14], 1)
        tmp_13 = tmp_14 = None
        tmp_16 = torch.nn.functional.batch_norm(tmp_15, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
        tmp_15 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_17 = torch.prelu(tmp_16, tmp_0)
        tmp_16 = tmp_0 = None
        tmp_18 = torch.nn.functional.adaptive_avg_pool2d(tmp_17, 1)
        tmp_19 = tmp_18.view(32, 128)
        tmp_18 = None
        tmp_20 = torch.nn.functional.linear(tmp_19, tmp_7, tmp_6)
        tmp_19 = tmp_7 = tmp_6 = None
        tmp_21 = torch.nn.functional.relu(tmp_20, inplace=True)
        tmp_20 = None
        tmp_22 = torch.nn.functional.linear(tmp_21, tmp_9, tmp_8)
        tmp_21 = tmp_9 = tmp_8 = None
        tmp_23 = torch.sigmoid(tmp_22)
        tmp_22 = None
        tmp_24 = tmp_23.view(32, 128, 1, 1)
        tmp_23 = None
        tmp_25 = tmp_17 * tmp_24
        tmp_17 = tmp_24 = None
        tmp_26 = in_12 + tmp_25
        tmp_25 = None
        return (tmp_26,)