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
        tmp_12 = in_12
        tmp_13 = torch.prelu(in_13, tmp_6)
        tmp_6 = None
        tmp_14 = torch.conv2d(tmp_13, tmp_11, None, (1, 1), (1, 1), (1, 1), 128)
        tmp_11 = None
        tmp_15 = torch.conv2d(tmp_13, tmp_12, None, (1, 1), (4, 4), (4, 4), 128)
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
        tmp_21 = tmp_20.view(1, 128)
        tmp_20 = None
        tmp_22 = torch.nn.functional.linear(tmp_21, tmp_8, tmp_7)
        tmp_21 = tmp_8 = tmp_7 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_10, tmp_9)
        tmp_23 = tmp_10 = tmp_9 = None
        tmp_25 = torch.sigmoid(tmp_24)
        tmp_24 = None
        tmp_26 = tmp_25.view(1, 128, 1, 1)
        tmp_25 = None
        tmp_27 = tmp_19 * tmp_26
        tmp_19 = tmp_26 = None
        return (tmp_27,)