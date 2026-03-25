import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19, in_20, in_21, in_22, in_23):
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
        tmp_21 = torch.nn.functional.dropout(in_21, 0.1, False, False)
        tmp_22 = in_23 * tmp_21
        tmp_21 = None
        tmp_23 = in_22 + tmp_22
        tmp_22 = None
        tmp_24 = tmp_23.flatten(2)
        tmp_23 = None
        tmp_25 = tmp_24.transpose(1, 2)
        tmp_24 = None
        tmp_26 = torch.nn.functional.layer_norm(tmp_25, (448,), tmp_5, tmp_4, 1e-12)
        tmp_5 = tmp_4 = None
        tmp_27 = torch.nn.functional.linear(tmp_26, tmp_16, tmp_15)
        tmp_26 = tmp_16 = tmp_15 = None
        tmp_28 = tmp_27.reshape(128, 49, 8, -1)
        tmp_27 = None
        tmp_29 = tmp_28.split([32, 32, 128], dim=3)
        tmp_28 = None
        tmp_30 = tmp_29[0]
        tmp_31 = tmp_29[1]
        tmp_32 = tmp_29[2]
        tmp_29 = None
        tmp_33 = tmp_30.permute(0, 2, 1, 3)
        tmp_30 = None
        tmp_34 = tmp_31.permute(0, 2, 1, 3)
        tmp_31 = None
        tmp_35 = tmp_32.permute(0, 2, 1, 3)
        tmp_32 = None
        tmp_36 = tmp_12.to(device(type='cuda', index=0))
        tmp_12 = None
        tmp_37 = tmp_34.transpose(-2, -1)
        tmp_34 = None
        tmp_38 = torch.matmul(tmp_33, tmp_37)
        tmp_33 = tmp_37 = None
        tmp_39 = tmp_38 * 0.1767766952966369
        tmp_38 = None
        tmp_40 = tmp_39 + tmp_36
        tmp_39 = None
        tmp_41 = tmp_40.softmax(dim=-1)
        tmp_40 = None
        tmp_42 = torch.matmul(tmp_41, tmp_35)
        tmp_41 = tmp_35 = None
        tmp_43 = tmp_42.transpose(1, 2)
        tmp_42 = None
        tmp_44 = tmp_43.reshape(128, 49, 1024)
        tmp_43 = None
        tmp_45 = torch.nn.functional.linear(tmp_44, tmp_14, tmp_13)
        tmp_44 = tmp_14 = tmp_13 = None
        tmp_46 = tmp_17.unsqueeze(0)
        tmp_17 = None
        tmp_47 = tmp_46.unsqueeze(0)
        tmp_46 = None
        tmp_48 = tmp_47 * tmp_45
        tmp_47 = tmp_45 = None
        tmp_49 = tmp_25 + tmp_48
        tmp_25 = tmp_48 = None
        tmp_50 = tmp_18.unsqueeze(0)
        tmp_18 = None
        tmp_51 = tmp_50.unsqueeze(0)
        tmp_50 = None
        tmp_52 = torch.nn.functional.layer_norm(tmp_49, (448,), tmp_7, tmp_6, 1e-12)
        tmp_7 = tmp_6 = None
        tmp_53 = torch.nn.functional.linear(tmp_52, tmp_9, tmp_8)
        tmp_52 = tmp_9 = tmp_8 = None
        tmp_54 = torch.nn.functional.gelu(tmp_53)
        tmp_53 = None
        tmp_55 = torch.nn.functional.dropout(tmp_54, 0.1, False, False)
        tmp_54 = None
        tmp_56 = torch.nn.functional.linear(tmp_55, tmp_11, tmp_10)
        tmp_55 = tmp_11 = tmp_10 = None
        tmp_57 = torch.nn.functional.dropout(tmp_56, 0.1, False, False)
        tmp_56 = None
        tmp_58 = tmp_51 * tmp_57
        tmp_51 = tmp_57 = None
        tmp_59 = tmp_49 + tmp_58
        tmp_49 = tmp_58 = None
        tmp_60 = torch.nn.functional.layer_norm(tmp_59, (448,), tmp_20, tmp_19, 1e-12)
        tmp_59 = tmp_20 = tmp_19 = None
        tmp_61 = tmp_60.mean(-2)
        tmp_62 = torch.nn.functional.linear(tmp_61, tmp_1, tmp_0)
        tmp_61 = tmp_1 = tmp_0 = None
        tmp_63 = tmp_60.mean(-2)
        tmp_60 = None
        tmp_64 = torch.nn.functional.linear(tmp_63, tmp_3, tmp_2)
        tmp_63 = tmp_3 = tmp_2 = None
        tmp_65 = tmp_62 + tmp_64
        tmp_66 = tmp_65 / 2
        tmp_65 = None
        return (tmp_36, tmp_62, tmp_64, tmp_66)