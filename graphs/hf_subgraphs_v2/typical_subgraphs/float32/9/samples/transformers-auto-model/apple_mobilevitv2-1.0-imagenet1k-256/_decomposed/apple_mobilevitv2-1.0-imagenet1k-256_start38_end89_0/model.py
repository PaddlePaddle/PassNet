import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, in_0):
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
        tmp_27 = torch.nn.functional.silu(in_0, inplace=False)
        tmp_28 = torch.conv2d(tmp_27, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
        tmp_27 = tmp_0 = None
        tmp_29 = torch.nn.functional.unfold(tmp_28, kernel_size=(2, 2), stride=(2, 2))
        tmp_28 = None
        tmp_30 = tmp_29.reshape(1, 128, 4, -1)
        tmp_29 = None
        tmp_31 = torch.nn.functional.group_norm(tmp_30, 1, tmp_14, tmp_13, 1e-05)
        tmp_14 = tmp_13 = None
        tmp_32 = torch.conv2d(tmp_31, tmp_6, tmp_5, (1, 1), (0, 0), (1, 1), 1)
        tmp_31 = tmp_6 = tmp_5 = None
        tmp_33 = torch.functional.split(tmp_32, split_size_or_sections=[1, 128, 128], dim=1)
        tmp_32 = None
        tmp_34 = tmp_33[0]
        tmp_35 = tmp_33[1]
        tmp_36 = tmp_33[2]
        tmp_33 = None
        tmp_37 = torch.nn.functional.softmax(tmp_34, dim=-1)
        tmp_34 = None
        tmp_38 = torch.nn.functional.dropout(tmp_37, 0.0, False, False)
        tmp_37 = None
        tmp_39 = tmp_35 * tmp_38
        tmp_35 = tmp_38 = None
        tmp_40 = torch.sum(tmp_39, dim=-1, keepdim=True)
        tmp_39 = None
        tmp_41 = torch.nn.functional.relu(tmp_36)
        tmp_42 = tmp_40.expand_as(tmp_36)
        tmp_40 = tmp_36 = None
        tmp_43 = tmp_41 * tmp_42
        tmp_41 = tmp_42 = None
        tmp_44 = torch.conv2d(tmp_43, tmp_4, tmp_3, (1, 1), (0, 0), (1, 1), 1)
        tmp_43 = tmp_4 = tmp_3 = None
        tmp_45 = tmp_44 + tmp_30
        tmp_44 = tmp_30 = None
        tmp_46 = torch.nn.functional.group_norm(tmp_45, 1, tmp_12, tmp_11, 1e-05)
        tmp_12 = tmp_11 = None
        tmp_47 = torch.conv2d(tmp_46, tmp_8, tmp_7, (1, 1), (0, 0), (1, 1), 1)
        tmp_46 = tmp_8 = tmp_7 = None
        tmp_48 = torch.nn.functional.silu(tmp_47, inplace=False)
        tmp_47 = None
        tmp_49 = torch.nn.functional.dropout(tmp_48, 0.0, False, False)
        tmp_48 = None
        tmp_50 = torch.conv2d(tmp_49, tmp_10, tmp_9, (1, 1), (0, 0), (1, 1), 1)
        tmp_49 = tmp_10 = tmp_9 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.0, False, False)
        tmp_50 = None
        tmp_52 = tmp_51 + tmp_45
        tmp_51 = tmp_45 = None
        tmp_53 = torch.nn.functional.group_norm(tmp_52, 1, tmp_26, tmp_25, 1e-05)
        tmp_26 = tmp_25 = None
        tmp_54 = torch.conv2d(tmp_53, tmp_18, tmp_17, (1, 1), (0, 0), (1, 1), 1)
        tmp_53 = tmp_18 = tmp_17 = None
        tmp_55 = torch.functional.split(tmp_54, split_size_or_sections=[1, 128, 128], dim=1)
        tmp_54 = None
        tmp_56 = tmp_55[0]
        tmp_57 = tmp_55[1]
        tmp_58 = tmp_55[2]
        tmp_55 = None
        tmp_59 = torch.nn.functional.softmax(tmp_56, dim=-1)
        tmp_56 = None
        tmp_60 = torch.nn.functional.dropout(tmp_59, 0.0, False, False)
        tmp_59 = None
        tmp_61 = tmp_57 * tmp_60
        tmp_57 = tmp_60 = None
        tmp_62 = torch.sum(tmp_61, dim=-1, keepdim=True)
        tmp_61 = None
        tmp_63 = torch.nn.functional.relu(tmp_58)
        tmp_64 = tmp_62.expand_as(tmp_58)
        tmp_62 = tmp_58 = None
        tmp_65 = tmp_63 * tmp_64
        tmp_63 = tmp_64 = None
        tmp_66 = torch.conv2d(tmp_65, tmp_16, tmp_15, (1, 1), (0, 0), (1, 1), 1)
        tmp_65 = tmp_16 = tmp_15 = None
        tmp_67 = tmp_66 + tmp_52
        tmp_66 = tmp_52 = None
        tmp_68 = torch.nn.functional.group_norm(tmp_67, 1, tmp_24, tmp_23, 1e-05)
        tmp_24 = tmp_23 = None
        tmp_69 = torch.conv2d(tmp_68, tmp_20, tmp_19, (1, 1), (0, 0), (1, 1), 1)
        tmp_68 = tmp_20 = tmp_19 = None
        tmp_70 = torch.nn.functional.silu(tmp_69, inplace=False)
        tmp_69 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.0, False, False)
        tmp_70 = None
        tmp_72 = torch.conv2d(tmp_71, tmp_22, tmp_21, (1, 1), (0, 0), (1, 1), 1)
        tmp_71 = tmp_22 = tmp_21 = None
        tmp_73 = torch.nn.functional.dropout(tmp_72, 0.0, False, False)
        tmp_72 = None
        tmp_74 = tmp_73 + tmp_67
        tmp_73 = tmp_67 = None
        tmp_75 = torch.nn.functional.group_norm(tmp_74, 1, tmp_2, tmp_1, 1e-05)
        tmp_74 = tmp_2 = tmp_1 = None
        tmp_76 = tmp_75.reshape(1, 512, 256)
        tmp_75 = None
        tmp_77 = torch.nn.functional.fold(tmp_76, output_size=(32, 32), kernel_size=(2, 2), stride=(2, 2))
        tmp_76 = None
        return (tmp_77,)