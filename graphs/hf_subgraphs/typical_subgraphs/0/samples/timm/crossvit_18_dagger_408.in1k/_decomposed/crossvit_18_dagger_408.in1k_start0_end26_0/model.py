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
        tmp_17 = in_17
        tmp_18 = in_18
        tmp_19 = in_19
        tmp_20 = in_20
        tmp_21 = torch.nn.functional.interpolate(tmp_20, size=(408, 408), mode='bicubic', align_corners=False)
        tmp_22 = torch.conv2d(tmp_21, tmp_5, tmp_4, (4, 4), (3, 3), (1, 1), 1)
        tmp_21 = tmp_5 = tmp_4 = None
        tmp_23 = torch.nn.functional.relu(tmp_22, inplace=True)
        tmp_22 = None
        tmp_24 = torch.conv2d(tmp_23, tmp_7, tmp_6, (3, 3), (0, 0), (1, 1), 1)
        tmp_23 = tmp_7 = tmp_6 = None
        tmp_25 = torch.nn.functional.relu(tmp_24, inplace=True)
        tmp_24 = None
        tmp_26 = torch.conv2d(tmp_25, tmp_9, tmp_8, (1, 1), (1, 1), (1, 1), 1)
        tmp_25 = tmp_9 = tmp_8 = None
        tmp_27 = tmp_26.flatten(2)
        tmp_26 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = tmp_16.expand(1, -1, -1)
        tmp_16 = None
        tmp_30 = torch.cat((tmp_29, tmp_28), dim=1)
        tmp_29 = tmp_28 = None
        tmp_31 = tmp_30 + tmp_18
        tmp_30 = tmp_18 = None
        tmp_32 = torch.nn.functional.dropout(tmp_31, 0.0, False, False)
        tmp_31 = None
        tmp_33 = torch.nn.functional.interpolate(tmp_20, size=(384, 384), mode='bicubic', align_corners=False)
        tmp_20 = None
        tmp_34 = torch.conv2d(tmp_33, tmp_11, tmp_10, (4, 4), (3, 3), (1, 1), 1)
        tmp_33 = tmp_11 = tmp_10 = None
        tmp_35 = torch.nn.functional.relu(tmp_34, inplace=True)
        tmp_34 = None
        tmp_36 = torch.conv2d(tmp_35, tmp_13, tmp_12, (2, 2), (1, 1), (1, 1), 1)
        tmp_35 = tmp_13 = tmp_12 = None
        tmp_37 = torch.nn.functional.relu(tmp_36, inplace=True)
        tmp_36 = None
        tmp_38 = torch.conv2d(tmp_37, tmp_15, tmp_14, (2, 2), (1, 1), (1, 1), 1)
        tmp_37 = tmp_15 = tmp_14 = None
        tmp_39 = tmp_38.flatten(2)
        tmp_38 = None
        tmp_40 = tmp_39.transpose(1, 2)
        tmp_39 = None
        tmp_41 = tmp_17.expand(1, -1, -1)
        tmp_17 = None
        tmp_42 = torch.cat((tmp_41, tmp_40), dim=1)
        tmp_41 = tmp_40 = None
        tmp_43 = tmp_42 + tmp_19
        tmp_42 = tmp_19 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.0, False, False)
        tmp_43 = None
        tmp_45 = torch.nn.functional.layer_norm(tmp_32, (224,), tmp_3, tmp_2, 1e-06)
        tmp_3 = tmp_2 = None
        tmp_46 = torch.nn.functional.linear(tmp_45, tmp_1, tmp_0)
        tmp_45 = tmp_1 = tmp_0 = None
        return (tmp_46, tmp_32, tmp_44)