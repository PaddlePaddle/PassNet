import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, w_20, w_21, in_0, in_1, in_2, in_3):
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
        tmp_22 = torch.nn.functional.linear(in_0, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_23 = tmp_22.view(4, -1, 4, 16)
        tmp_22 = None
        tmp_24 = tmp_23.transpose(1, 2)
        tmp_23 = None
        tmp_25 = torch.nn.functional.normalize(in_3, dim=-1)
        tmp_26 = torch.nn.functional.normalize(in_2, dim=-1)
        tmp_27 = tmp_26.transpose(-2, -1)
        tmp_26 = None
        tmp_28 = tmp_25 @ tmp_27
        tmp_25 = tmp_27 = None
        tmp_29 = torch.clamp(tmp_11, max=4.605170185988092)
        tmp_11 = None
        tmp_30 = tmp_29.exp()
        tmp_29 = None
        tmp_31 = tmp_28 * tmp_30
        tmp_28 = tmp_30 = None
        tmp_32 = torch.nn.functional.linear(tmp_4, tmp_7, tmp_6)
        tmp_4 = tmp_7 = tmp_6 = None
        tmp_33 = torch.nn.functional.relu(tmp_32, inplace=True)
        tmp_32 = None
        tmp_34 = torch.nn.functional.linear(tmp_33, tmp_8, None)
        tmp_33 = tmp_8 = None
        tmp_35 = tmp_34.view(-1, 4)
        tmp_34 = None
        tmp_36 = tmp_5.view(-1)
        tmp_5 = None
        tmp_37 = tmp_35[tmp_36]
        tmp_35 = tmp_36 = None
        tmp_38 = tmp_37.view(4, 4, -1)
        tmp_37 = None
        tmp_39 = tmp_38.permute(2, 0, 1)
        tmp_38 = None
        tmp_40 = tmp_39.contiguous()
        tmp_39 = None
        tmp_41 = torch.sigmoid(tmp_40)
        tmp_40 = None
        tmp_42 = 16 * tmp_41
        tmp_41 = None
        tmp_43 = tmp_42.unsqueeze(0)
        tmp_42 = None
        tmp_44 = tmp_31 + tmp_43
        tmp_31 = tmp_43 = None
        tmp_45 = torch.nn.functional.softmax(tmp_44, dim=-1)
        tmp_44 = None
        tmp_46 = torch.nn.functional.dropout(tmp_45, 0.0, False, False)
        tmp_45 = None
        tmp_47 = torch.matmul(tmp_46, tmp_24)
        tmp_46 = tmp_24 = None
        tmp_48 = tmp_47.permute(0, 2, 1, 3)
        tmp_47 = None
        tmp_49 = tmp_48.contiguous()
        tmp_48 = None
        tmp_50 = tmp_49.view((4, 4, 64))
        tmp_49 = None
        tmp_51 = torch.nn.functional.linear(tmp_50, tmp_3, tmp_2)
        tmp_50 = tmp_3 = tmp_2 = None
        tmp_52 = torch.nn.functional.dropout(tmp_51, 0.0, False, False)
        tmp_51 = None
        tmp_53 = tmp_52.view(-1, 2, 2, 64)
        tmp_52 = None
        tmp_54 = tmp_53.view(-1, 2, 2, 2, 2, 64)
        tmp_53 = None
        tmp_55 = tmp_54.permute(0, 1, 3, 2, 4, 5)
        tmp_54 = None
        tmp_56 = tmp_55.contiguous()
        tmp_55 = None
        tmp_57 = tmp_56.view(-1, 4, 4, 64)
        tmp_56 = None
        tmp_58 = tmp_57.view(1, 16, 64)
        tmp_57 = None
        tmp_59 = torch.nn.functional.layer_norm(tmp_58, (64,), tmp_17, tmp_16, 1e-05)
        tmp_58 = tmp_17 = tmp_16 = None
        tmp_60 = in_1 + tmp_59
        tmp_59 = None
        tmp_61 = torch.nn.functional.linear(tmp_60, tmp_13, tmp_12)
        tmp_13 = tmp_12 = None
        tmp_62 = torch.nn.functional.gelu(tmp_61)
        tmp_61 = None
        tmp_63 = torch.nn.functional.linear(tmp_62, tmp_19, tmp_18)
        tmp_62 = tmp_19 = tmp_18 = None
        tmp_64 = torch.nn.functional.dropout(tmp_63, 0.0, False, False)
        tmp_63 = None
        tmp_65 = torch.nn.functional.layer_norm(tmp_64, (64,), tmp_15, tmp_14, 1e-05)
        tmp_64 = tmp_15 = tmp_14 = None
        tmp_66 = tmp_60 + tmp_65
        tmp_60 = tmp_65 = None
        tmp_67 = torch.nn.functional.layer_norm(tmp_66, (64,), tmp_21, tmp_20, 1e-05)
        tmp_66 = tmp_21 = tmp_20 = None
        tmp_68 = tmp_67.transpose(1, 2)
        tmp_67 = None
        tmp_69 = torch.adaptive_avg_pool1d(tmp_68, 1)
        tmp_68 = None
        tmp_70 = torch.flatten(tmp_69, 1)
        tmp_69 = None
        tmp_71 = torch.nn.functional.linear(tmp_70, tmp_1, tmp_0)
        tmp_70 = tmp_1 = tmp_0 = None
        return (tmp_71,)