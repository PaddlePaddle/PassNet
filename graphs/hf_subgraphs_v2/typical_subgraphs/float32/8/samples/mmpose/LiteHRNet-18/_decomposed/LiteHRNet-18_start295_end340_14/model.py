import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
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
        tmp_12 = torch.nn.functional.adaptive_avg_pool2d(in_15, 1)
        tmp_13 = torch.conv2d(tmp_12, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_12 = tmp_1 = tmp_0 = None
        tmp_14 = torch.nn.functional.relu(tmp_13, inplace=True)
        tmp_13 = None
        tmp_15 = torch.conv2d(tmp_14, tmp_3, tmp_2, (1, 1), (0, 0), (1, 1), 1)
        tmp_14 = tmp_3 = tmp_2 = None
        tmp_16 = torch.sigmoid(tmp_15)
        tmp_15 = None
        tmp_17 = in_15 * tmp_16
        tmp_16 = None
        tmp_18 = torch.nn.functional.adaptive_avg_pool2d(in_16, 1)
        tmp_19 = torch.conv2d(tmp_18, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_18 = tmp_5 = tmp_4 = None
        tmp_20 = torch.nn.functional.relu(tmp_19, inplace=True)
        tmp_19 = None
        tmp_21 = torch.conv2d(tmp_20, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_20 = tmp_7 = tmp_6 = None
        tmp_22 = torch.sigmoid(tmp_21)
        tmp_21 = None
        tmp_23 = in_16 * tmp_22
        tmp_22 = None
        tmp_24 = torch.nn.functional.adaptive_avg_pool2d(in_17, 1)
        tmp_25 = torch.conv2d(tmp_24, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_24 = tmp_9 = tmp_8 = None
        tmp_26 = torch.nn.functional.relu(tmp_25, inplace=True)
        tmp_25 = None
        tmp_27 = torch.conv2d(tmp_26, tmp_11, tmp_10, (1, 1), (0, 0), (1, 1), 1)
        tmp_26 = tmp_11 = tmp_10 = None
        tmp_28 = torch.sigmoid(tmp_27)
        tmp_27 = None
        tmp_29 = in_17 * tmp_28
        tmp_28 = None
        tmp_30 = torch.cat([in_13, tmp_17], dim=1)
        tmp_17 = None
        tmp_31 = torch.cat([in_14, tmp_23], dim=1)
        tmp_23 = None
        tmp_32 = torch.cat([in_12, tmp_29], dim=1)
        tmp_29 = None
        tmp_33 = tmp_30.view(512, 2, 20, 64, 48)
        tmp_30 = None
        tmp_34 = torch.transpose(tmp_33, 1, 2)
        tmp_33 = None
        tmp_35 = tmp_34.contiguous()
        tmp_34 = None
        tmp_36 = tmp_35.view(512, 40, 64, 48)
        tmp_35 = None
        tmp_37 = tmp_31.view(512, 2, 40, 32, 24)
        tmp_31 = None
        tmp_38 = torch.transpose(tmp_37, 1, 2)
        tmp_37 = None
        tmp_39 = tmp_38.contiguous()
        tmp_38 = None
        tmp_40 = tmp_39.view(512, 80, 32, 24)
        tmp_39 = None
        tmp_41 = tmp_32.view(512, 2, 80, 16, 12)
        tmp_32 = None
        tmp_42 = torch.transpose(tmp_41, 1, 2)
        tmp_41 = None
        tmp_43 = tmp_42.contiguous()
        tmp_42 = None
        tmp_44 = tmp_43.view(512, 160, 16, 12)
        tmp_43 = None
        tmp_45 = tmp_36.chunk(2, dim=1)
        tmp_36 = None
        tmp_46 = tmp_45[0]
        tmp_47 = tmp_45[1]
        tmp_45 = None
        tmp_48 = tmp_40.chunk(2, dim=1)
        tmp_40 = None
        tmp_49 = tmp_48[0]
        tmp_50 = tmp_48[1]
        tmp_48 = None
        tmp_51 = tmp_44.chunk(2, dim=1)
        tmp_44 = None
        tmp_52 = tmp_51[0]
        tmp_53 = tmp_51[1]
        tmp_51 = None
        tmp_54 = torch.nn.functional.adaptive_avg_pool2d(tmp_47, (16, 12))
        tmp_55 = torch.nn.functional.adaptive_avg_pool2d(tmp_50, (16, 12))
        tmp_56 = torch.cat([tmp_54, tmp_55, tmp_53], dim=1)
        tmp_54 = tmp_55 = None
        return (tmp_56, tmp_46, tmp_49, tmp_52, tmp_47, tmp_50, tmp_53)