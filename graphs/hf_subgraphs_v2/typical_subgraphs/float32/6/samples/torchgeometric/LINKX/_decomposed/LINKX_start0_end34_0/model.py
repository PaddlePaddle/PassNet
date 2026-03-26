import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24, in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32, in_33, in_34, in_35):
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
        tmp_34 = in_34
        tmp_35 = in_35
        tmp_36 = tmp_0[1]
        tmp_37 = tmp_0[0]
        tmp_0 = None
        tmp_38 = tmp_6.index_select(-2, tmp_37)
        tmp_6 = tmp_37 = None
        tmp_39 = tmp_36.view((-1, 1))
        tmp_36 = None
        tmp_40 = tmp_39.expand_as(tmp_38)
        tmp_39 = None
        tmp_41 = tmp_38.new_zeros((1000, 128))
        tmp_42 = tmp_41.scatter_add_(0, tmp_40, tmp_38)
        tmp_41 = tmp_40 = tmp_38 = None
        tmp_43 = tmp_42 + tmp_5
        tmp_42 = tmp_5 = None
        tmp_44 = torch.nn.functional.linear(tmp_43, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_45 = tmp_43 + tmp_44
        tmp_43 = tmp_44 = None
        tmp_46 = torch.nn.functional.linear(tmp_35, tmp_34, tmp_33)
        tmp_35 = tmp_34 = tmp_33 = None
        tmp_47 = torch.nn.functional.dropout(tmp_46, p=0.0, training=False)
        tmp_46 = None
        tmp_48 = tmp_45 + tmp_47
        tmp_45 = None
        tmp_49 = torch.nn.functional.linear(tmp_47, tmp_4, tmp_3)
        tmp_47 = tmp_4 = tmp_3 = None
        tmp_50 = tmp_48 + tmp_49
        tmp_48 = tmp_49 = None
        tmp_51 = tmp_50.relu_()
        tmp_50 = None
        tmp_52 = torch.nn.functional.linear(tmp_51, tmp_8, tmp_7)
        tmp_51 = tmp_8 = tmp_7 = None
        tmp_53 = torch.nn.functional.relu(tmp_52, inplace=False)
        tmp_52 = None
        tmp_54 = torch.nn.functional.batch_norm(tmp_53, tmp_17, tmp_18, tmp_20, tmp_19, False, 0.1, 1e-05)
        tmp_53 = tmp_17 = tmp_18 = tmp_20 = tmp_19 = None
        tmp_55 = torch.nn.functional.dropout(tmp_54, p=0.0, training=False)
        tmp_54 = None
        tmp_56 = torch.nn.functional.linear(tmp_55, tmp_10, tmp_9)
        tmp_55 = tmp_10 = tmp_9 = None
        tmp_57 = torch.nn.functional.relu(tmp_56, inplace=False)
        tmp_56 = None
        tmp_58 = torch.nn.functional.batch_norm(tmp_57, tmp_21, tmp_22, tmp_24, tmp_23, False, 0.1, 1e-05)
        tmp_57 = tmp_21 = tmp_22 = tmp_24 = tmp_23 = None
        tmp_59 = torch.nn.functional.dropout(tmp_58, p=0.0, training=False)
        tmp_58 = None
        tmp_60 = torch.nn.functional.linear(tmp_59, tmp_12, tmp_11)
        tmp_59 = tmp_12 = tmp_11 = None
        tmp_61 = torch.nn.functional.relu(tmp_60, inplace=False)
        tmp_60 = None
        tmp_62 = torch.nn.functional.batch_norm(tmp_61, tmp_25, tmp_26, tmp_28, tmp_27, False, 0.1, 1e-05)
        tmp_61 = tmp_25 = tmp_26 = tmp_28 = tmp_27 = None
        tmp_63 = torch.nn.functional.dropout(tmp_62, p=0.0, training=False)
        tmp_62 = None
        tmp_64 = torch.nn.functional.linear(tmp_63, tmp_14, tmp_13)
        tmp_63 = tmp_14 = tmp_13 = None
        tmp_65 = torch.nn.functional.relu(tmp_64, inplace=False)
        tmp_64 = None
        tmp_66 = torch.nn.functional.batch_norm(tmp_65, tmp_29, tmp_30, tmp_32, tmp_31, False, 0.1, 1e-05)
        tmp_65 = tmp_29 = tmp_30 = tmp_32 = tmp_31 = None
        tmp_67 = torch.nn.functional.dropout(tmp_66, p=0.0, training=False)
        tmp_66 = None
        tmp_68 = torch.nn.functional.linear(tmp_67, tmp_16, tmp_15)
        tmp_67 = tmp_16 = tmp_15 = None
        tmp_69 = torch.nn.functional.dropout(tmp_68, p=0.0, training=False)
        tmp_68 = None
        return (tmp_69,)