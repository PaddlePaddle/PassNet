import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21):
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
        tmp_17 = in_17
        tmp_18 = in_18
        tmp_19 = torch.prelu(in_21, tmp_5)
        tmp_5 = None
        tmp_20 = torch.conv2d(tmp_19, tmp_10, None, (1, 1), (1, 1), (1, 1), 64)
        tmp_10 = None
        tmp_21 = torch.conv2d(tmp_19, tmp_11, None, (1, 1), (4, 4), (4, 4), 64)
        tmp_19 = tmp_11 = None
        tmp_22 = torch.cat([tmp_20, tmp_21], 1)
        tmp_20 = tmp_21 = None
        tmp_23 = torch.nn.functional.batch_norm(tmp_22, tmp_1, tmp_2, tmp_4, tmp_3, False, 0.1, 0.001)
        tmp_22 = tmp_1 = tmp_2 = tmp_4 = tmp_3 = None
        tmp_24 = torch.prelu(tmp_23, tmp_0)
        tmp_23 = tmp_0 = None
        tmp_25 = torch.nn.functional.adaptive_avg_pool2d(tmp_24, 1)
        tmp_26 = tmp_25.view(1, 128)
        tmp_25 = None
        tmp_27 = torch.nn.functional.linear(tmp_26, tmp_7, tmp_6)
        tmp_26 = tmp_7 = tmp_6 = None
        tmp_28 = torch.nn.functional.relu(tmp_27, inplace=True)
        tmp_27 = None
        tmp_29 = torch.nn.functional.linear(tmp_28, tmp_9, tmp_8)
        tmp_28 = tmp_9 = tmp_8 = None
        tmp_30 = torch.sigmoid(tmp_29)
        tmp_29 = None
        tmp_31 = tmp_30.view(1, 128, 1, 1)
        tmp_30 = None
        tmp_32 = tmp_24 * tmp_31
        tmp_24 = tmp_31 = None
        tmp_33 = in_19 + tmp_32
        tmp_32 = None
        tmp_34 = torch.cat([in_20, tmp_33], 1)
        tmp_33 = None
        tmp_35 = torch.nn.functional.batch_norm(tmp_34, tmp_12, tmp_13, tmp_15, tmp_14, False, 0.1, 0.001)
        tmp_34 = tmp_12 = tmp_13 = tmp_15 = tmp_14 = None
        tmp_36 = torch.prelu(tmp_35, tmp_16)
        tmp_35 = tmp_16 = None
        tmp_37 = torch.conv2d(tmp_36, tmp_18, tmp_17, (1, 1), (0, 0), (1, 1), 1)
        tmp_36 = tmp_18 = tmp_17 = None
        return (tmp_37,)