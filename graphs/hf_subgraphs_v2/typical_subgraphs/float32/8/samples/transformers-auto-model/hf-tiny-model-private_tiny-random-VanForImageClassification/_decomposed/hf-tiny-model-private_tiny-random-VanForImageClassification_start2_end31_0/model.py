import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28):
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
        tmp_25 = in_25
        tmp_26 = in_26
        tmp_27 = in_27
        tmp_28 = torch.nn.functional.batch_norm(in_28, tmp_22, tmp_23, tmp_25, tmp_24, False, 0.1, 1e-05)
        tmp_22 = tmp_23 = tmp_25 = tmp_24 = None
        tmp_29 = torch.conv2d(tmp_28, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_9 = tmp_8 = None
        tmp_30 = torch.nn.functional.gelu(tmp_29)
        tmp_29 = None
        tmp_31 = torch.conv2d(tmp_30, tmp_3, tmp_2, (1, 1), (2, 2), (1, 1), 16)
        tmp_3 = tmp_2 = None
        tmp_32 = torch.conv2d(tmp_31, tmp_1, tmp_0, (1, 1), (9, 9), (3, 3), 16)
        tmp_31 = tmp_1 = tmp_0 = None
        tmp_33 = torch.conv2d(tmp_32, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)
        tmp_32 = tmp_5 = tmp_4 = None
        tmp_34 = tmp_30 * tmp_33
        tmp_30 = tmp_33 = None
        tmp_35 = torch.conv2d(tmp_34, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_34 = tmp_7 = tmp_6 = None
        tmp_36 = tmp_35 + tmp_28
        tmp_35 = tmp_28 = None
        tmp_37 = tmp_10.unsqueeze(-1)
        tmp_10 = None
        tmp_38 = tmp_37.unsqueeze(-1)
        tmp_37 = None
        tmp_39 = tmp_38 * tmp_36
        tmp_38 = tmp_36 = None
        tmp_40 = in_28 + tmp_39
        tmp_39 = None
        tmp_41 = torch.nn.functional.batch_norm(tmp_40, tmp_18, tmp_19, tmp_21, tmp_20, False, 0.1, 1e-05)
        tmp_18 = tmp_19 = tmp_21 = tmp_20 = None
        tmp_42 = torch.conv2d(tmp_41, tmp_14, tmp_13, (1, 1), (0, 0), (1, 1), 1)
        tmp_41 = tmp_14 = tmp_13 = None
        tmp_43 = torch.conv2d(tmp_42, tmp_12, tmp_11, (1, 1), (1, 1), (1, 1), 128)
        tmp_42 = tmp_12 = tmp_11 = None
        tmp_44 = torch.nn.functional.gelu(tmp_43)
        tmp_43 = None
        tmp_45 = torch.nn.functional.dropout(tmp_44, 0.0, False, False)
        tmp_44 = None
        tmp_46 = torch.conv2d(tmp_45, tmp_16, tmp_15, (1, 1), (0, 0), (1, 1), 1)
        tmp_45 = tmp_16 = tmp_15 = None
        tmp_47 = torch.nn.functional.dropout(tmp_46, 0.0, False, False)
        tmp_46 = None
        tmp_48 = tmp_17.unsqueeze(-1)
        tmp_17 = None
        tmp_49 = tmp_48.unsqueeze(-1)
        tmp_48 = None
        tmp_50 = tmp_49 * tmp_47
        tmp_49 = tmp_47 = None
        tmp_51 = tmp_40 + tmp_50
        tmp_40 = tmp_50 = None
        tmp_52 = tmp_51.flatten(2)
        tmp_51 = None
        tmp_53 = tmp_52.transpose(1, 2)
        tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (16,), tmp_27, tmp_26, 1e-06)
        tmp_53 = tmp_27 = tmp_26 = None
        tmp_55 = tmp_54.view(512, 56, 56, 16)
        tmp_54 = None
        tmp_56 = tmp_55.permute(0, 3, 1, 2)
        tmp_55 = None
        return (tmp_56,)