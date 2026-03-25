import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26):
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
        tmp_21 = in_21
        tmp_22 = in_22
        tmp_23 = in_23
        tmp_24 = in_24
        tmp_25 = in_26.view(8, 608, 48)
        tmp_26 = tmp_25.unsqueeze(1)
        tmp_25 = None
        tmp_27 = torch.conv2d(in_26, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_7 = tmp_6 = None
        tmp_28 = tmp_27.view(8, 1, 48)
        tmp_27 = None
        tmp_29 = torch.nn.functional.softmax(tmp_28, 2, _stacklevel=5)
        tmp_28 = None
        tmp_30 = tmp_29.unsqueeze(-1)
        tmp_29 = None
        tmp_31 = torch.matmul(tmp_26, tmp_30)
        tmp_26 = tmp_30 = None
        tmp_32 = tmp_31.view(8, 608, 1, 1)
        tmp_31 = None
        tmp_33 = torch.conv2d(tmp_32, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
        tmp_32 = tmp_1 = tmp_0 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (38, 1, 1), tmp_3, tmp_2, 1e-05)
        tmp_33 = tmp_3 = tmp_2 = None
        tmp_35 = torch.nn.functional.relu(tmp_34, inplace=True)
        tmp_34 = None
        tmp_36 = torch.conv2d(tmp_35, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_35 = tmp_5 = tmp_4 = None
        tmp_37 = in_26 + tmp_36
        tmp_36 = None
        tmp_37 += in_25
        tmp_38 = tmp_37
        tmp_37 = None
        tmp_39 = torch.nn.functional.relu(tmp_38, inplace=True)
        tmp_38 = None
        tmp_40 = torch.conv_transpose2d(tmp_39, tmp_8, None, (2, 2), (1, 1), (0, 0), 16, (1, 1))
        tmp_39 = tmp_8 = None
        tmp_41 = torch.nn.functional.batch_norm(tmp_40, tmp_9, tmp_10, tmp_12, tmp_11, False, 0.1, 1e-05)
        tmp_40 = tmp_9 = tmp_10 = tmp_12 = tmp_11 = None
        tmp_42 = torch.nn.functional.relu(tmp_41, inplace=True)
        tmp_41 = None
        tmp_43 = torch.conv_transpose2d(tmp_42, tmp_13, None, (2, 2), (1, 1), (0, 0), 16, (1, 1))
        tmp_42 = tmp_13 = None
        tmp_44 = torch.nn.functional.batch_norm(tmp_43, tmp_14, tmp_15, tmp_17, tmp_16, False, 0.1, 1e-05)
        tmp_43 = tmp_14 = tmp_15 = tmp_17 = tmp_16 = None
        tmp_45 = torch.nn.functional.relu(tmp_44, inplace=True)
        tmp_44 = None
        tmp_46 = torch.conv_transpose2d(tmp_45, tmp_18, None, (2, 2), (1, 1), (0, 0), 16, (1, 1))
        tmp_45 = tmp_18 = None
        tmp_47 = torch.nn.functional.batch_norm(tmp_46, tmp_19, tmp_20, tmp_22, tmp_21, False, 0.1, 1e-05)
        tmp_46 = tmp_19 = tmp_20 = tmp_22 = tmp_21 = None
        tmp_48 = torch.nn.functional.relu(tmp_47, inplace=True)
        tmp_47 = None
        tmp_49 = torch.conv2d(tmp_48, tmp_24, tmp_23, (1, 1), (0, 0), (1, 1), 1)
        tmp_48 = tmp_24 = tmp_23 = None
        return (tmp_49,)