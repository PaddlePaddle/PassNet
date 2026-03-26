import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35, in_36, in_37, in_38):
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
        tmp_28 = in_28
        tmp_29 = in_29
        tmp_30 = in_30
        tmp_31 = in_31
        tmp_32 = in_32
        tmp_33 = in_33
        tmp_34 = torch.nn.functional.linear(in_37, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_35 = tmp_34.view(1, -1, 4, 32)
        tmp_34 = None
        tmp_36 = tmp_35.transpose(1, 2)
        tmp_35 = None
        tmp_37 = in_35.transpose(-1, -2)
        tmp_38 = torch.matmul(in_38, tmp_37)
        tmp_37 = None
        tmp_39 = tmp_38 / 5.656854249492381
        tmp_38 = None
        tmp_40 = tmp_39 + in_34
        tmp_39 = None
        tmp_41 = torch.nn.functional.softmax(tmp_40, dim=-1)
        tmp_40 = None
        tmp_42 = torch.nn.functional.dropout(tmp_41, 0.1, False, False)
        tmp_41 = None
        tmp_43 = torch.matmul(tmp_42, tmp_36)
        tmp_42 = tmp_36 = None
        tmp_44 = tmp_43.permute(0, 2, 1, 3)
        tmp_43 = None
        tmp_45 = tmp_44.contiguous()
        tmp_44 = None
        tmp_46 = tmp_45.view((1, 512, 128))
        tmp_45 = None
        tmp_47 = torch.nn.functional.linear(tmp_46, tmp_3, tmp_2)
        tmp_46 = tmp_3 = tmp_2 = None
        tmp_48 = tmp_47 + in_36
        tmp_47 = None
        tmp_49 = tmp_48 * tmp_1
        tmp_48 = tmp_1 = None
        tmp_50 = tmp_49 + tmp_0
        tmp_49 = tmp_0 = None
        tmp_51 = torch.nn.functional.linear(tmp_50, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_52 = torch.nn.functional.relu(tmp_51, inplace=False)
        tmp_51 = None
        tmp_53 = torch.nn.functional.linear(tmp_52, tmp_11, tmp_10)
        tmp_52 = tmp_11 = tmp_10 = None
        tmp_54 = tmp_53 + tmp_50
        tmp_53 = tmp_50 = None
        tmp_55 = tmp_54 * tmp_9
        tmp_54 = tmp_9 = None
        tmp_56 = tmp_55 + tmp_8
        tmp_55 = tmp_8 = None
        tmp_57 = torch.nn.functional.linear(tmp_56, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_58 = torch.nn.functional.relu(tmp_57, inplace=False)
        tmp_57 = None
        tmp_59 = torch.nn.functional.linear(tmp_58, tmp_17, tmp_16)
        tmp_58 = tmp_17 = tmp_16 = None
        tmp_60 = tmp_59 + tmp_56
        tmp_59 = tmp_56 = None
        tmp_61 = tmp_60 * tmp_15
        tmp_60 = tmp_15 = None
        tmp_62 = tmp_61 + tmp_14
        tmp_61 = tmp_14 = None
        tmp_63 = torch.nn.functional.linear(tmp_62, tmp_19, tmp_18)
        tmp_19 = tmp_18 = None
        tmp_64 = torch.nn.functional.relu(tmp_63, inplace=False)
        tmp_63 = None
        tmp_65 = torch.nn.functional.linear(tmp_64, tmp_23, tmp_22)
        tmp_64 = tmp_23 = tmp_22 = None
        tmp_66 = tmp_65 + tmp_62
        tmp_65 = tmp_62 = None
        tmp_67 = tmp_66 * tmp_21
        tmp_66 = tmp_21 = None
        tmp_68 = tmp_67 + tmp_20
        tmp_67 = tmp_20 = None
        tmp_69 = torch.nn.functional.linear(tmp_68, tmp_25, tmp_24)
        tmp_25 = tmp_24 = None
        tmp_70 = torch.nn.functional.relu(tmp_69, inplace=False)
        tmp_69 = None
        tmp_71 = torch.nn.functional.linear(tmp_70, tmp_33, tmp_32)
        tmp_70 = tmp_33 = tmp_32 = None
        tmp_72 = tmp_71 + tmp_68
        tmp_71 = tmp_68 = None
        tmp_73 = tmp_72 * tmp_27
        tmp_72 = tmp_27 = None
        tmp_74 = tmp_73 + tmp_26
        tmp_73 = tmp_26 = None
        tmp_75 = torch.nn.functional.linear(tmp_74, tmp_31, tmp_30)
        tmp_74 = tmp_31 = tmp_30 = None
        tmp_76 = torch.nn.functional.dropout(tmp_75, 0.0, False, False)
        tmp_75 = None
        tmp_77 = tmp_76 + in_37
        tmp_76 = None
        tmp_78 = tmp_77 * tmp_29
        tmp_77 = tmp_29 = None
        tmp_79 = tmp_78 + tmp_28
        tmp_78 = tmp_28 = None
        tmp_80 = torch.tensor(1000)
        tmp_80 = None
        tmp_81 = tmp_79[slice(None, None, None), 0]
        return (tmp_79, tmp_81)