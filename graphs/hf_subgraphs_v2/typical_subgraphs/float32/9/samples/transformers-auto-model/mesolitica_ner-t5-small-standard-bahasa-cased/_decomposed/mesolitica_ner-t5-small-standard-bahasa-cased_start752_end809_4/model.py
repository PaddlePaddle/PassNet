import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3, in_4):
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
        tmp_10 = in_0.view(1, -1, 512)
        tmp_11 = torch.nn.functional.linear(tmp_10, tmp_0, None)
        tmp_10 = tmp_0 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
        tmp_11 = None
        tmp_13 = in_3 + tmp_12
        tmp_12 = None
        tmp_14 = in_1[-1]
        tmp_15 = tmp_14 + 1
        tmp_14 = tmp_15 = None
        tmp_16 = tmp_13.to(torch.float32)
        tmp_17 = tmp_16.pow(2)
        tmp_16 = None
        tmp_18 = tmp_17.mean(-1, keepdim=True)
        tmp_17 = None
        tmp_19 = tmp_18 + 1e-06
        tmp_18 = None
        tmp_20 = torch.rsqrt(tmp_19)
        tmp_19 = None
        tmp_21 = tmp_13 * tmp_20
        tmp_20 = None
        tmp_22 = tmp_5 * tmp_21
        tmp_5 = tmp_21 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_3, None)
        tmp_22 = tmp_3 = None
        tmp_24 = tmp_23.view(1, -1, 8, 64)
        tmp_23 = None
        tmp_25 = tmp_24.transpose(1, 2)
        tmp_24 = None
        tmp_26 = torch.nn.functional.linear(in_2, tmp_1, None)
        tmp_1 = None
        tmp_27 = torch.nn.functional.linear(in_2, tmp_4, None)
        tmp_4 = None
        tmp_28 = tmp_26.view(1, -1, 8, 64)
        tmp_26 = None
        tmp_29 = tmp_28.transpose(1, 2)
        tmp_28 = None
        tmp_30 = tmp_27.view(1, -1, 8, 64)
        tmp_27 = None
        tmp_31 = tmp_30.transpose(1, 2)
        tmp_30 = None
        tmp_32 = tmp_29.transpose(3, 2)
        tmp_29 = None
        tmp_33 = torch.matmul(tmp_25, tmp_32)
        tmp_25 = tmp_32 = None
        tmp_33 += in_4
        tmp_34 = tmp_33
        tmp_33 = None
        tmp_35 = tmp_34.float()
        tmp_36 = torch.nn.functional.softmax(tmp_35, dim=-1)
        tmp_35 = None
        tmp_37 = tmp_36.type_as(tmp_34)
        tmp_36 = tmp_34 = None
        tmp_38 = torch.nn.functional.dropout(tmp_37, p=0.1, training=False)
        tmp_37 = None
        tmp_39 = torch.matmul(tmp_38, tmp_31)
        tmp_38 = tmp_31 = None
        tmp_40 = tmp_39.transpose(1, 2)
        tmp_39 = None
        tmp_41 = tmp_40.contiguous()
        tmp_40 = None
        tmp_42 = tmp_41.view(1, -1, 512)
        tmp_41 = None
        tmp_43 = torch.nn.functional.linear(tmp_42, tmp_2, None)
        tmp_42 = tmp_2 = None
        tmp_44 = torch.nn.functional.dropout(tmp_43, 0.1, False, False)
        tmp_43 = None
        tmp_45 = tmp_13 + tmp_44
        tmp_13 = tmp_44 = None
        tmp_46 = tmp_45.to(torch.float32)
        tmp_47 = tmp_46.pow(2)
        tmp_46 = None
        tmp_48 = tmp_47.mean(-1, keepdim=True)
        tmp_47 = None
        tmp_49 = tmp_48 + 1e-06
        tmp_48 = None
        tmp_50 = torch.rsqrt(tmp_49)
        tmp_49 = None
        tmp_51 = tmp_45 * tmp_50
        tmp_50 = None
        tmp_52 = tmp_8 * tmp_51
        tmp_8 = tmp_51 = None
        tmp_53 = torch.nn.functional.linear(tmp_52, tmp_6, None)
        tmp_52 = tmp_6 = None
        tmp_54 = torch.nn.functional.relu(tmp_53, inplace=False)
        tmp_53 = None
        tmp_55 = torch.nn.functional.dropout(tmp_54, 0.1, False, False)
        tmp_54 = None
        tmp_56 = torch.nn.functional.linear(tmp_55, tmp_7, None)
        tmp_55 = tmp_7 = None
        tmp_57 = torch.nn.functional.dropout(tmp_56, 0.1, False, False)
        tmp_56 = None
        tmp_58 = tmp_45 + tmp_57
        tmp_45 = tmp_57 = None
        tmp_59 = tmp_58.to(torch.float32)
        tmp_60 = tmp_59.pow(2)
        tmp_59 = None
        tmp_61 = tmp_60.mean(-1, keepdim=True)
        tmp_60 = None
        tmp_62 = tmp_61 + 1e-06
        tmp_61 = None
        tmp_63 = torch.rsqrt(tmp_62)
        tmp_62 = None
        tmp_64 = tmp_58 * tmp_63
        tmp_58 = tmp_63 = None
        tmp_65 = tmp_9 * tmp_64
        tmp_9 = tmp_64 = None
        tmp_66 = torch.nn.functional.dropout(tmp_65, 0.1, False, False)
        tmp_65 = None
        return (tmp_66,)