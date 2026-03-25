import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24):
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
        tmp_20 = torch.nn.functional.relu(in_22, inplace=False)
        tmp_21 = torch.nn.functional.dropout(tmp_20, p=0.0, training=False)
        tmp_20 = None
        tmp_22 = torch.nn.functional.linear(tmp_21, tmp_1, tmp_0)
        tmp_21 = tmp_1 = tmp_0 = None
        tmp_23 = torch.nn.functional.dropout(tmp_22, p=0.1, training=False)
        tmp_22 = None
        tmp_24 = in_20 + tmp_23
        tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (256,), tmp_3, tmp_2, 1e-05)
        tmp_24 = tmp_3 = tmp_2 = None
        tmp_26 = tmp_25 + in_24
        tmp_27 = torch.nn.functional.linear(tmp_26, tmp_17, tmp_16)
        tmp_17 = tmp_16 = None
        tmp_28 = tmp_27 * 0.1767766952966369
        tmp_27 = None
        tmp_29 = torch.nn.functional.linear(tmp_26, tmp_13, tmp_12)
        tmp_26 = tmp_13 = tmp_12 = None
        tmp_30 = tmp_29.view(1, -1, 8, 32)
        tmp_29 = None
        tmp_31 = tmp_30.transpose(1, 2)
        tmp_30 = None
        tmp_32 = tmp_31.contiguous()
        tmp_31 = None
        tmp_33 = torch.nn.functional.linear(tmp_25, tmp_19, tmp_18)
        tmp_19 = tmp_18 = None
        tmp_34 = tmp_33.view(1, -1, 8, 32)
        tmp_33 = None
        tmp_35 = tmp_34.transpose(1, 2)
        tmp_34 = None
        tmp_36 = tmp_35.contiguous()
        tmp_35 = None
        tmp_37 = tmp_28.view(1, 100, 8, 32)
        tmp_28 = None
        tmp_38 = tmp_37.transpose(1, 2)
        tmp_37 = None
        tmp_39 = tmp_38.contiguous()
        tmp_38 = None
        tmp_40 = tmp_39.view(8, -1, 32)
        tmp_39 = None
        tmp_41 = tmp_32.view(8, -1, 32)
        tmp_32 = None
        tmp_42 = tmp_36.view(8, -1, 32)
        tmp_36 = None
        tmp_43 = tmp_41.transpose(1, 2)
        tmp_41 = None
        tmp_44 = torch.bmm(tmp_40, tmp_43)
        tmp_40 = tmp_43 = None
        tmp_45 = torch.nn.functional.softmax(tmp_44, dim=-1)
        tmp_44 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, p=0.0, training=False)
        tmp_45 = None
        tmp_47 = torch.bmm(tmp_46, tmp_42)
        tmp_46 = tmp_42 = None
        tmp_48 = tmp_47.view(1, 8, 100, 32)
        tmp_47 = None
        tmp_49 = tmp_48.transpose(1, 2)
        tmp_48 = None
        tmp_50 = tmp_49.reshape(1, 100, 256)
        tmp_49 = None
        tmp_51 = torch.nn.functional.linear(tmp_50, tmp_15, tmp_14)
        tmp_50 = tmp_15 = tmp_14 = None
        tmp_52 = torch.nn.functional.dropout(tmp_51, p=0.1, training=False)
        tmp_51 = None
        tmp_53 = tmp_25 + tmp_52
        tmp_25 = tmp_52 = None
        tmp_54 = torch.nn.functional.layer_norm(tmp_53, (256,), tmp_11, tmp_10, 1e-05)
        tmp_53 = tmp_11 = tmp_10 = None
        tmp_55 = tmp_54 + in_24
        tmp_56 = in_21 + in_23
        tmp_57 = torch.nn.functional.linear(tmp_55, tmp_7, tmp_6)
        tmp_55 = tmp_7 = tmp_6 = None
        tmp_58 = tmp_57 * 0.1767766952966369
        tmp_57 = None
        tmp_59 = torch.nn.functional.linear(tmp_56, tmp_5, tmp_4)
        tmp_56 = tmp_5 = tmp_4 = None
        tmp_60 = tmp_59.view(1, -1, 8, 32)
        tmp_59 = None
        tmp_61 = tmp_60.transpose(1, 2)
        tmp_60 = None
        tmp_62 = tmp_61.contiguous()
        tmp_61 = None
        tmp_63 = torch.nn.functional.linear(in_21, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_64 = tmp_63.view(1, -1, 8, 32)
        tmp_63 = None
        tmp_65 = tmp_64.transpose(1, 2)
        tmp_64 = None
        tmp_66 = tmp_65.contiguous()
        tmp_65 = None
        tmp_67 = tmp_58.view(1, 100, 8, 32)
        tmp_58 = None
        tmp_68 = tmp_67.transpose(1, 2)
        tmp_67 = None
        tmp_69 = tmp_68.contiguous()
        tmp_68 = None
        tmp_70 = tmp_69.view(8, -1, 32)
        tmp_69 = None
        tmp_71 = tmp_62.view(8, -1, 32)
        tmp_62 = None
        tmp_72 = tmp_66.view(8, -1, 32)
        tmp_66 = None
        tmp_73 = tmp_71.transpose(1, 2)
        tmp_71 = None
        tmp_74 = torch.bmm(tmp_70, tmp_73)
        tmp_70 = tmp_73 = None
        tmp_75 = torch.nn.functional.softmax(tmp_74, dim=-1)
        tmp_74 = None
        tmp_76 = torch.nn.functional.dropout(tmp_75, p=0.0, training=False)
        tmp_75 = None
        tmp_77 = torch.bmm(tmp_76, tmp_72)
        tmp_76 = tmp_72 = None
        tmp_78 = tmp_77.view(1, 8, 100, 32)
        tmp_77 = None
        tmp_79 = tmp_78.transpose(1, 2)
        tmp_78 = None
        tmp_80 = tmp_79.reshape(1, 100, 256)
        tmp_79 = None
        return (tmp_80, tmp_54)