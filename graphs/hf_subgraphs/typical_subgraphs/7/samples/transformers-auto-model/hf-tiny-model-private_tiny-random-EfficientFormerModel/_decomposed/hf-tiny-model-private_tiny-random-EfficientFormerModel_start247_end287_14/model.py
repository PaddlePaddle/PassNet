import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18, in_19):
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
        tmp_17 = torch.nn.functional.dropout(in_17, 0.1, False, False)
        tmp_18 = in_19 * tmp_17
        tmp_17 = None
        tmp_19 = in_18 + tmp_18
        tmp_18 = None
        tmp_20 = tmp_19.flatten(2)
        tmp_19 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (448,), tmp_1, tmp_0, 1e-12)
        tmp_1 = tmp_0 = None
        tmp_23 = torch.nn.functional.linear(tmp_22, tmp_12, tmp_11)
        tmp_22 = tmp_12 = tmp_11 = None
        tmp_24 = tmp_23.reshape(256, 49, 8, -1)
        tmp_23 = None
        tmp_25 = tmp_24.split([32, 32, 128], dim=3)
        tmp_24 = None
        tmp_26 = tmp_25[0]
        tmp_27 = tmp_25[1]
        tmp_28 = tmp_25[2]
        tmp_25 = None
        tmp_29 = tmp_26.permute(0, 2, 1, 3)
        tmp_26 = None
        tmp_30 = tmp_27.permute(0, 2, 1, 3)
        tmp_27 = None
        tmp_31 = tmp_28.permute(0, 2, 1, 3)
        tmp_28 = None
        tmp_32 = tmp_8.to(device(type='cuda', index=0))
        tmp_8 = None
        tmp_33 = tmp_30.transpose(-2, -1)
        tmp_30 = None
        tmp_34 = torch.matmul(tmp_29, tmp_33)
        tmp_29 = tmp_33 = None
        tmp_35 = tmp_34 * 0.1767766952966369
        tmp_34 = None
        tmp_36 = tmp_35 + tmp_32
        tmp_35 = None
        tmp_37 = tmp_36.softmax(dim=-1)
        tmp_36 = None
        tmp_38 = torch.matmul(tmp_37, tmp_31)
        tmp_37 = tmp_31 = None
        tmp_39 = tmp_38.transpose(1, 2)
        tmp_38 = None
        tmp_40 = tmp_39.reshape(256, 49, 1024)
        tmp_39 = None
        tmp_41 = torch.nn.functional.linear(tmp_40, tmp_10, tmp_9)
        tmp_40 = tmp_10 = tmp_9 = None
        tmp_42 = tmp_13.unsqueeze(0)
        tmp_13 = None
        tmp_43 = tmp_42.unsqueeze(0)
        tmp_42 = None
        tmp_44 = tmp_43 * tmp_41
        tmp_43 = tmp_41 = None
        tmp_45 = tmp_21 + tmp_44
        tmp_21 = tmp_44 = None
        tmp_46 = tmp_14.unsqueeze(0)
        tmp_14 = None
        tmp_47 = tmp_46.unsqueeze(0)
        tmp_46 = None
        tmp_48 = torch.nn.functional.layer_norm(tmp_45, (448,), tmp_3, tmp_2, 1e-12)
        tmp_3 = tmp_2 = None
        tmp_49 = torch.nn.functional.linear(tmp_48, tmp_5, tmp_4)
        tmp_48 = tmp_5 = tmp_4 = None
        tmp_50 = torch.nn.functional.gelu(tmp_49)
        tmp_49 = None
        tmp_51 = torch.nn.functional.dropout(tmp_50, 0.1, False, False)
        tmp_50 = None
        tmp_52 = torch.nn.functional.linear(tmp_51, tmp_7, tmp_6)
        tmp_51 = tmp_7 = tmp_6 = None
        tmp_53 = torch.nn.functional.dropout(tmp_52, 0.1, False, False)
        tmp_52 = None
        tmp_54 = tmp_47 * tmp_53
        tmp_47 = tmp_53 = None
        tmp_55 = tmp_45 + tmp_54
        tmp_45 = tmp_54 = None
        tmp_56 = torch.nn.functional.layer_norm(tmp_55, (448,), tmp_16, tmp_15, 1e-12)
        tmp_55 = tmp_16 = tmp_15 = None
        return (tmp_32, tmp_56)