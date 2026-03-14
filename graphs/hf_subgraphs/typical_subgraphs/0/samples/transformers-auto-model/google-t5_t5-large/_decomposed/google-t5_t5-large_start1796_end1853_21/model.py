import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14):
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
        tmp_11 = in_12.view(1, -1, 1024)
        tmp_12 = torch.nn.functional.linear(tmp_11, tmp_1, None)
        tmp_11 = tmp_1 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.1, False, False)
        tmp_12 = None
        tmp_14 = in_13 + tmp_13
        tmp_13 = None
        tmp_15 = in_11[-1]
        tmp_16 = tmp_15 + 1
        tmp_15 = tmp_16 = None
        tmp_17 = tmp_14.to(torch.float32)
        tmp_18 = tmp_17.pow(2)
        tmp_17 = None
        tmp_19 = tmp_18.mean(-1, keepdim=True)
        tmp_18 = None
        tmp_20 = tmp_19 + 1e-06
        tmp_19 = None
        tmp_21 = torch.rsqrt(tmp_20)
        tmp_20 = None
        tmp_22 = tmp_14 * tmp_21
        tmp_21 = None
        tmp_23 = tmp_6 * tmp_22
        tmp_6 = tmp_22 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_4, None)
        tmp_23 = tmp_4 = None
        tmp_25 = tmp_24.view(1, -1, 16, 64)
        tmp_24 = None
        tmp_26 = tmp_25.transpose(1, 2)
        tmp_25 = None
        tmp_27 = torch.nn.functional.linear(tmp_0, tmp_2, None)
        tmp_2 = None
        tmp_28 = torch.nn.functional.linear(tmp_0, tmp_5, None)
        tmp_0 = tmp_5 = None
        tmp_29 = tmp_27.view(1, -1, 16, 64)
        tmp_27 = None
        tmp_30 = tmp_29.transpose(1, 2)
        tmp_29 = None
        tmp_31 = tmp_28.view(1, -1, 16, 64)
        tmp_28 = None
        tmp_32 = tmp_31.transpose(1, 2)
        tmp_31 = None
        tmp_33 = tmp_30.transpose(3, 2)
        tmp_34 = torch.matmul(tmp_26, tmp_33)
        tmp_26 = tmp_33 = None
        tmp_34 += in_14
        tmp_35 = tmp_34
        tmp_34 = None
        tmp_36 = tmp_35.float()
        tmp_37 = torch.nn.functional.softmax(tmp_36, dim=-1)
        tmp_36 = None
        tmp_38 = tmp_37.type_as(tmp_35)
        tmp_37 = tmp_35 = None
        tmp_39 = torch.nn.functional.dropout(tmp_38, p=0.1, training=False)
        tmp_38 = None
        tmp_40 = torch.matmul(tmp_39, tmp_32)
        tmp_39 = None
        tmp_41 = tmp_40.transpose(1, 2)
        tmp_40 = None
        tmp_42 = tmp_41.contiguous()
        tmp_41 = None
        tmp_43 = tmp_42.view(1, -1, 1024)
        tmp_42 = None
        tmp_44 = torch.nn.functional.linear(tmp_43, tmp_3, None)
        tmp_43 = tmp_3 = None
        tmp_45 = torch.nn.functional.dropout(tmp_44, 0.1, False, False)
        tmp_44 = None
        tmp_46 = tmp_14 + tmp_45
        tmp_14 = tmp_45 = None
        tmp_47 = tmp_46.to(torch.float32)
        tmp_48 = tmp_47.pow(2)
        tmp_47 = None
        tmp_49 = tmp_48.mean(-1, keepdim=True)
        tmp_48 = None
        tmp_50 = tmp_49 + 1e-06
        tmp_49 = None
        tmp_51 = torch.rsqrt(tmp_50)
        tmp_50 = None
        tmp_52 = tmp_46 * tmp_51
        tmp_51 = None
        tmp_53 = tmp_9 * tmp_52
        tmp_9 = tmp_52 = None
        tmp_54 = torch.nn.functional.linear(tmp_53, tmp_7, None)
        tmp_53 = tmp_7 = None
        tmp_55 = torch.nn.functional.relu(tmp_54, inplace=False)
        tmp_54 = None
        tmp_56 = torch.nn.functional.dropout(tmp_55, 0.1, False, False)
        tmp_55 = None
        tmp_57 = torch.nn.functional.linear(tmp_56, tmp_8, None)
        tmp_56 = tmp_8 = None
        tmp_58 = torch.nn.functional.dropout(tmp_57, 0.1, False, False)
        tmp_57 = None
        tmp_59 = tmp_46 + tmp_58
        tmp_46 = tmp_58 = None
        tmp_60 = tmp_59.to(torch.float32)
        tmp_61 = tmp_60.pow(2)
        tmp_60 = None
        tmp_62 = tmp_61.mean(-1, keepdim=True)
        tmp_61 = None
        tmp_63 = tmp_62 + 1e-06
        tmp_62 = None
        tmp_64 = torch.rsqrt(tmp_63)
        tmp_63 = None
        tmp_65 = tmp_59 * tmp_64
        tmp_59 = tmp_64 = None
        tmp_66 = tmp_10 * tmp_65
        tmp_10 = tmp_65 = None
        tmp_67 = torch.nn.functional.dropout(tmp_66, 0.1, False, False)
        tmp_66 = None
        return (tmp_30, tmp_32, tmp_67)