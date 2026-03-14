import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.nn.functional.adaptive_avg_pool2d(in_10, 1)
        tmp_9 = torch.conv2d(tmp_8, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_8 = tmp_1 = tmp_0 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace=True)
        tmp_9 = None
        tmp_11 = torch.conv2d(tmp_10, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_10 = tmp_3 = tmp_2 = None
        tmp_12 = torch.sigmoid(tmp_11)
        tmp_11 = None
        tmp_13 = in_10 * tmp_12
        tmp_12 = None
        tmp_14 = torch.nn.functional.adaptive_avg_pool2d(in_11, 1)
        tmp_15 = torch.conv2d(tmp_14, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_14 = tmp_5 = tmp_4 = None
        tmp_16 = torch.nn.functional.relu(tmp_15, inplace=True)
        tmp_15 = None
        tmp_17 = torch.conv2d(tmp_16, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_16 = tmp_7 = tmp_6 = None
        tmp_18 = torch.sigmoid(tmp_17)
        tmp_17 = None
        tmp_19 = in_11 * tmp_18
        tmp_18 = None
        tmp_20 = torch.cat([in_8, tmp_13], dim=1)
        tmp_13 = None
        tmp_21 = torch.cat([in_9, tmp_19], dim=1)
        tmp_19 = None
        tmp_22 = tmp_20.view(64, 2, 20, 64, 48)
        tmp_20 = None
        tmp_23 = torch.transpose(tmp_22, 1, 2)
        tmp_22 = None
        tmp_24 = tmp_23.contiguous()
        tmp_23 = None
        tmp_25 = tmp_24.view(64, 40, 64, 48)
        tmp_24 = None
        tmp_26 = tmp_21.view(64, 2, 40, 32, 24)
        tmp_21 = None
        tmp_27 = torch.transpose(tmp_26, 1, 2)
        tmp_26 = None
        tmp_28 = tmp_27.contiguous()
        tmp_27 = None
        tmp_29 = tmp_28.view(64, 80, 32, 24)
        tmp_28 = None
        tmp_30 = tmp_25.chunk(2, dim=1)
        tmp_25 = None
        tmp_31 = tmp_30[0]
        tmp_32 = tmp_30[1]
        tmp_30 = None
        tmp_33 = tmp_29.chunk(2, dim=1)
        tmp_29 = None
        tmp_34 = tmp_33[0]
        tmp_35 = tmp_33[1]
        tmp_33 = None
        tmp_36 = torch.nn.functional.adaptive_avg_pool2d(tmp_32, (32, 24))
        tmp_37 = torch.cat([tmp_36, tmp_35], dim=1)
        tmp_36 = None
        return (tmp_37, tmp_31, tmp_34, tmp_32, tmp_35)