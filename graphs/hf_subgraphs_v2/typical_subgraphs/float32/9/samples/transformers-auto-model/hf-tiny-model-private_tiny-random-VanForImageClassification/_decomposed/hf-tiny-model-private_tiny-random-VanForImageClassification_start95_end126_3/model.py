import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, in_0):
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
        tmp_12 = w_12
        tmp_13 = w_13
        tmp_14 = w_14
        tmp_15 = w_15
        tmp_16 = w_16
        tmp_17 = w_17
        tmp_18 = w_18
        tmp_19 = w_19
        tmp_20 = w_20
        tmp_21 = w_21
        tmp_22 = w_22
        tmp_23 = w_23
        tmp_24 = w_24
        tmp_25 = w_25
        tmp_26 = w_26
        tmp_27 = w_27
        tmp_28 = w_28
        tmp_29 = w_29
        tmp_30 = torch.nn.functional.batch_norm(in_0, tmp_24, tmp_25, tmp_27, tmp_26, False, 0.1, 1e-05)
        tmp_24 = tmp_25 = tmp_27 = tmp_26 = None
        tmp_31 = torch.conv2d(tmp_30, tmp_11, tmp_10, (1, 1), (0, 0), (1, 1), 1)
        tmp_11 = tmp_10 = None
        tmp_32 = torch.nn.functional.gelu(tmp_31)
        tmp_31 = None
        tmp_33 = torch.conv2d(tmp_32, tmp_5, tmp_4, (1, 1), (2, 2), (1, 1), 128)
        tmp_5 = tmp_4 = None
        tmp_34 = torch.conv2d(tmp_33, tmp_3, tmp_2, (1, 1), (9, 9), (3, 3), 128)
        tmp_33 = tmp_3 = tmp_2 = None
        tmp_35 = torch.conv2d(tmp_34, tmp_7, tmp_6, (1, 1), (0, 0), (1, 1), 1)
        tmp_34 = tmp_7 = tmp_6 = None
        tmp_36 = tmp_32 * tmp_35
        tmp_32 = tmp_35 = None
        tmp_37 = torch.conv2d(tmp_36, tmp_9, tmp_8, (1, 1), (0, 0), (1, 1), 1)
        tmp_36 = tmp_9 = tmp_8 = None
        tmp_38 = tmp_37 + tmp_30
        tmp_37 = tmp_30 = None
        tmp_39 = tmp_12.unsqueeze(-1)
        tmp_12 = None
        tmp_40 = tmp_39.unsqueeze(-1)
        tmp_39 = None
        tmp_41 = tmp_40 * tmp_38
        tmp_40 = tmp_38 = None
        tmp_42 = in_0 + tmp_41
        tmp_41 = None
        tmp_43 = torch.nn.functional.batch_norm(tmp_42, tmp_20, tmp_21, tmp_23, tmp_22, False, 0.1, 1e-05)
        tmp_20 = tmp_21 = tmp_23 = tmp_22 = None
        tmp_44 = torch.conv2d(tmp_43, tmp_16, tmp_15, (1, 1), (0, 0), (1, 1), 1)
        tmp_43 = tmp_16 = tmp_15 = None
        tmp_45 = torch.conv2d(tmp_44, tmp_14, tmp_13, (1, 1), (1, 1), (1, 1), 512)
        tmp_44 = tmp_14 = tmp_13 = None
        tmp_46 = torch.nn.functional.gelu(tmp_45)
        tmp_45 = None
        tmp_47 = torch.nn.functional.dropout(tmp_46, 0.0, False, False)
        tmp_46 = None
        tmp_48 = torch.conv2d(tmp_47, tmp_18, tmp_17, (1, 1), (0, 0), (1, 1), 1)
        tmp_47 = tmp_18 = tmp_17 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.0, False, False)
        tmp_48 = None
        tmp_50 = tmp_19.unsqueeze(-1)
        tmp_19 = None
        tmp_51 = tmp_50.unsqueeze(-1)
        tmp_50 = None
        tmp_52 = tmp_51 * tmp_49
        tmp_51 = tmp_49 = None
        tmp_53 = tmp_42 + tmp_52
        tmp_42 = tmp_52 = None
        tmp_54 = tmp_53.flatten(2)
        tmp_53 = None
        tmp_55 = tmp_54.transpose(1, 2)
        tmp_54 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (128,), tmp_29, tmp_28, 1e-06)
        tmp_55 = tmp_29 = tmp_28 = None
        tmp_57 = tmp_56.view(1, 7, 7, 128)
        tmp_56 = None
        tmp_58 = tmp_57.permute(0, 3, 1, 2)
        tmp_57 = None
        tmp_59 = tmp_58.mean(dim=[-2, -1])
        tmp_58 = None
        tmp_60 = torch.nn.functional.linear(tmp_59, tmp_1, tmp_0)
        tmp_59 = tmp_1 = tmp_0 = None
        return (tmp_60,)