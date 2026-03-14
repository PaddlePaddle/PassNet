import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19, in_0, in_1, in_2, in_3, in_4):
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
        tmp_20 = torch.nn.functional.linear(in_1, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_21 = tmp_20.view(16, -1, 24, 32)
        tmp_20 = None
        tmp_22 = tmp_21.transpose(1, 2)
        tmp_21 = None
        tmp_23 = torch.nn.functional.normalize(in_4, dim=-1)
        tmp_24 = torch.nn.functional.normalize(in_2, dim=-1)
        tmp_25 = tmp_24.transpose(-2, -1)
        tmp_24 = None
        tmp_26 = tmp_23 @ tmp_25
        tmp_23 = tmp_25 = None
        tmp_27 = torch.clamp(tmp_9, max=4.605170185988092)
        tmp_9 = None
        tmp_28 = tmp_27.exp()
        tmp_27 = None
        tmp_29 = tmp_26 * tmp_28
        tmp_26 = tmp_28 = None
        tmp_30 = torch.nn.functional.linear(tmp_2, tmp_5, tmp_4)
        tmp_2 = tmp_5 = tmp_4 = None
        tmp_31 = torch.nn.functional.relu(tmp_30, inplace=True)
        tmp_30 = None
        tmp_32 = torch.nn.functional.linear(tmp_31, tmp_6, None)
        tmp_31 = tmp_6 = None
        tmp_33 = tmp_32.view(-1, 24)
        tmp_32 = None
        tmp_34 = tmp_3.view(-1)
        tmp_3 = None
        tmp_35 = tmp_33[tmp_34]
        tmp_33 = tmp_34 = None
        tmp_36 = tmp_35.view(64, 64, -1)
        tmp_35 = None
        tmp_37 = tmp_36.permute(2, 0, 1)
        tmp_36 = None
        tmp_38 = tmp_37.contiguous()
        tmp_37 = None
        tmp_39 = torch.sigmoid(tmp_38)
        tmp_38 = None
        tmp_40 = 16 * tmp_39
        tmp_39 = None
        tmp_41 = tmp_40.unsqueeze(0)
        tmp_40 = None
        tmp_42 = tmp_29 + tmp_41
        tmp_29 = tmp_41 = None
        tmp_43 = tmp_42.view(1, 16, 24, 64, 64)
        tmp_42 = None
        tmp_44 = in_0.unsqueeze(1)
        tmp_45 = tmp_44.unsqueeze(0)
        tmp_44 = None
        tmp_46 = tmp_43 + tmp_45
        tmp_43 = tmp_45 = None
        tmp_47 = in_0.unsqueeze(1)
        tmp_48 = tmp_47.unsqueeze(0)
        tmp_47 = None
        tmp_49 = tmp_46 + tmp_48
        tmp_46 = tmp_48 = None
        tmp_50 = tmp_49.view(-1, 24, 64, 64)
        tmp_49 = None
        tmp_51 = torch.nn.functional.softmax(tmp_50, dim=-1)
        tmp_50 = None
        tmp_52 = torch.nn.functional.dropout(tmp_51, 0.0, False, False)
        tmp_51 = None
        tmp_53 = torch.matmul(tmp_52, tmp_22)
        tmp_52 = tmp_22 = None
        tmp_54 = tmp_53.permute(0, 2, 1, 3)
        tmp_53 = None
        tmp_55 = tmp_54.contiguous()
        tmp_54 = None
        tmp_56 = tmp_55.view((16, 64, 768))
        tmp_55 = None
        tmp_57 = torch.nn.functional.linear(tmp_56, tmp_1, tmp_0)
        tmp_56 = tmp_1 = tmp_0 = None
        tmp_58 = torch.nn.functional.dropout(tmp_57, 0.0, False, False)
        tmp_57 = None
        tmp_59 = tmp_58.view(-1, 8, 8, 768)
        tmp_58 = None
        tmp_60 = tmp_59.view(-1, 4, 4, 8, 8, 768)
        tmp_59 = None
        tmp_61 = tmp_60.permute(0, 1, 3, 2, 4, 5)
        tmp_60 = None
        tmp_62 = tmp_61.contiguous()
        tmp_61 = None
        tmp_63 = tmp_62.view(-1, 32, 32, 768)
        tmp_62 = None
        tmp_64 = torch.roll(tmp_63, shifts=(4, 4), dims=(1, 2))
        tmp_63 = None
        tmp_65 = tmp_64.view(1, 1024, 768)
        tmp_64 = None
        tmp_66 = torch.nn.functional.layer_norm(tmp_65, (768,), tmp_15, tmp_14, 1e-05)
        tmp_65 = tmp_15 = tmp_14 = None
        tmp_67 = in_3 + tmp_66
        tmp_66 = None
        tmp_68 = torch.nn.functional.linear(tmp_67, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_69 = torch.nn.functional.gelu(tmp_68)
        tmp_68 = None
        tmp_70 = torch.nn.functional.linear(tmp_69, tmp_17, tmp_16)
        tmp_69 = tmp_17 = tmp_16 = None
        tmp_71 = torch.nn.functional.dropout(tmp_70, 0.0, False, False)
        tmp_70 = None
        tmp_72 = torch.nn.functional.layer_norm(tmp_71, (768,), tmp_13, tmp_12, 1e-05)
        tmp_71 = tmp_13 = tmp_12 = None
        tmp_73 = tmp_67 + tmp_72
        tmp_67 = tmp_72 = None
        tmp_74 = torch.nn.functional.layer_norm(tmp_73, (768,), tmp_19, tmp_18, 1e-05)
        tmp_73 = tmp_19 = tmp_18 = None
        tmp_75 = tmp_74.transpose(1, 2)
        tmp_76 = torch.adaptive_avg_pool1d(tmp_75, 1)
        tmp_75 = None
        tmp_77 = torch.flatten(tmp_76, 1)
        tmp_76 = None
        return (tmp_74, tmp_77)