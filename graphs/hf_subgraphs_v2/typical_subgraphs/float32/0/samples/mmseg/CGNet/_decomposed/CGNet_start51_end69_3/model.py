import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20):
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
        tmp_17 = torch.prelu(in_19, tmp_5)
        tmp_5 = None
        tmp_18 = torch.conv2d(tmp_17, tmp_10, None, (1, 1), (1, 1), (1, 1), 32)
        tmp_10 = None
        tmp_19 = torch.conv2d(tmp_17, tmp_11, None, (1, 1), (2, 2), (2, 2), 32)
        tmp_17 = tmp_11 = None
        tmp_20 = torch.cat([tmp_18, tmp_19], 1)
        tmp_18 = tmp_19 = None
        tmp_21 = torch.nn.functional.batch_norm(tmp_20, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
        tmp_20 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_22 = torch.prelu(tmp_21, tmp_0)
        tmp_21 = tmp_0 = None
        tmp_23 = torch.nn.functional.adaptive_avg_pool2d(tmp_22, 1)
        tmp_24 = tmp_23.view(1, 64)
        tmp_23 = None
        tmp_25 = torch.nn.functional.linear(tmp_24, tmp_7, tmp_6)
        tmp_24 = tmp_7 = tmp_6 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace=True)
        tmp_25 = None
        tmp_27 = torch.nn.functional.linear(tmp_26, tmp_9, tmp_8)
        tmp_26 = tmp_9 = tmp_8 = None
        tmp_28 = torch.sigmoid(tmp_27)
        tmp_27 = None
        tmp_29 = tmp_28.view(1, 64, 1, 1)
        tmp_28 = None
        tmp_30 = tmp_22 * tmp_29
        tmp_22 = tmp_29 = None
        tmp_31 = in_18 + tmp_30
        tmp_30 = None
        tmp_32 = torch.cat([tmp_31, in_17, in_20], 1)
        tmp_31 = None
        tmp_33 = torch.nn.functional.batch_norm(tmp_32, tmp_12, tmp_13, tmp_15, tmp_14, False, 0.1, 0.001)
        tmp_32 = tmp_12 = tmp_13 = tmp_15 = tmp_14 = None
        tmp_34 = torch.prelu(tmp_33, tmp_16)
        tmp_33 = tmp_16 = None
        return (tmp_34,)