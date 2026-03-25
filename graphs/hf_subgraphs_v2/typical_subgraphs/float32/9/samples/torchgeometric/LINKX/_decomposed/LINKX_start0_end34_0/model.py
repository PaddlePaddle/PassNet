import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, w_22, w_23, w_24, w_25, w_26, w_27, w_28, w_29, w_30, w_31, w_32, w_33, in_1):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = w_10
        tmp_12 = w_11
        tmp_13 = w_12
        tmp_14 = w_13
        tmp_15 = w_14
        tmp_16 = w_15
        tmp_17 = w_16
        tmp_18 = w_17
        tmp_19 = w_18
        tmp_20 = w_19
        tmp_21 = w_20
        tmp_22 = w_21
        tmp_23 = w_22
        tmp_24 = w_23
        tmp_25 = w_24
        tmp_26 = w_25
        tmp_27 = w_26
        tmp_28 = w_27
        tmp_29 = w_28
        tmp_30 = w_29
        tmp_31 = w_30
        tmp_32 = w_31
        tmp_33 = w_32
        tmp_34 = w_33
        tmp_35 = in_1
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