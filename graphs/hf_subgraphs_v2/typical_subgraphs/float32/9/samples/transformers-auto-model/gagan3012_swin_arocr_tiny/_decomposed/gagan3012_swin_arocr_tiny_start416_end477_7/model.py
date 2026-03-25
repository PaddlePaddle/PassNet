import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, in_0, in_1, in_2, in_3, in_4):
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
        tmp_18 = torch.nn.functional.linear(in_1, tmp_8, tmp_7)
        tmp_8 = tmp_7 = None
        tmp_19 = tmp_18.view(64, -1, 12, 32)
        tmp_18 = None
        tmp_20 = tmp_19.transpose(1, 2)
        tmp_19 = None
        tmp_21 = torch.nn.functional.normalize(in_4, dim=-1)
        tmp_22 = torch.nn.functional.normalize(in_2, dim=-1)
        tmp_23 = tmp_22.transpose(-2, -1)
        tmp_22 = None
        tmp_24 = tmp_21 @ tmp_23
        tmp_21 = tmp_23 = None
        tmp_25 = torch.clamp(tmp_9, max=4.605170185988092)
        tmp_9 = None
        tmp_26 = tmp_25.exp()
        tmp_25 = None
        tmp_27 = tmp_24 * tmp_26
        tmp_24 = tmp_26 = None
        tmp_28 = torch.nn.functional.linear(tmp_2, tmp_5, tmp_4)
        tmp_2 = tmp_5 = tmp_4 = None
        tmp_29 = torch.nn.functional.relu(tmp_28, inplace=True)
        tmp_28 = None
        tmp_30 = torch.nn.functional.linear(tmp_29, tmp_6, None)
        tmp_29 = tmp_6 = None
        tmp_31 = tmp_30.view(-1, 12)
        tmp_30 = None
        tmp_32 = tmp_3.view(-1)
        tmp_3 = None
        tmp_33 = tmp_31[tmp_32]
        tmp_31 = tmp_32 = None
        tmp_34 = tmp_33.view(64, 64, -1)
        tmp_33 = None
        tmp_35 = tmp_34.permute(2, 0, 1)
        tmp_34 = None
        tmp_36 = tmp_35.contiguous()
        tmp_35 = None
        tmp_37 = torch.sigmoid(tmp_36)
        tmp_36 = None
        tmp_38 = 16 * tmp_37
        tmp_37 = None
        tmp_39 = tmp_38.unsqueeze(0)
        tmp_38 = None
        tmp_40 = tmp_27 + tmp_39
        tmp_27 = tmp_39 = None
        tmp_41 = tmp_40.view(1, 64, 12, 64, 64)
        tmp_40 = None
        tmp_42 = in_0.unsqueeze(1)
        tmp_43 = tmp_42.unsqueeze(0)
        tmp_42 = None
        tmp_44 = tmp_41 + tmp_43
        tmp_41 = tmp_43 = None
        tmp_45 = in_0.unsqueeze(1)
        tmp_46 = tmp_45.unsqueeze(0)
        tmp_45 = None
        tmp_47 = tmp_44 + tmp_46
        tmp_44 = tmp_46 = None
        tmp_48 = tmp_47.view(-1, 12, 64, 64)
        tmp_47 = None
        tmp_49 = torch.nn.functional.softmax(tmp_48, dim=-1)
        tmp_48 = None
        tmp_50 = torch.nn.functional.dropout(tmp_49, 0.0, False, False)
        tmp_49 = None
        tmp_51 = torch.matmul(tmp_50, tmp_20)
        tmp_50 = tmp_20 = None
        tmp_52 = tmp_51.permute(0, 2, 1, 3)
        tmp_51 = None
        tmp_53 = tmp_52.contiguous()
        tmp_52 = None
        tmp_54 = tmp_53.view((64, 64, 384))
        tmp_53 = None
        tmp_55 = torch.nn.functional.linear(tmp_54, tmp_1, tmp_0)
        tmp_54 = tmp_1 = tmp_0 = None
        tmp_56 = torch.nn.functional.dropout(tmp_55, 0.0, False, False)
        tmp_55 = None
        tmp_57 = tmp_56.view(-1, 8, 8, 384)
        tmp_56 = None
        tmp_58 = tmp_57.view(-1, 8, 8, 8, 8, 384)
        tmp_57 = None
        tmp_59 = tmp_58.permute(0, 1, 3, 2, 4, 5)
        tmp_58 = None
        tmp_60 = tmp_59.contiguous()
        tmp_59 = None
        tmp_61 = tmp_60.view(-1, 64, 64, 384)
        tmp_60 = None
        tmp_62 = torch.roll(tmp_61, shifts=(4, 4), dims=(1, 2))
        tmp_61 = None
        tmp_63 = tmp_62.view(1, 4096, 384)
        tmp_62 = None
        tmp_64 = torch.nn.functional.layer_norm(tmp_63, (384,), tmp_15, tmp_14, 1e-05)
        tmp_63 = tmp_15 = tmp_14 = None
        tmp_65 = in_3 + tmp_64
        tmp_64 = None
        tmp_66 = torch.nn.functional.linear(tmp_65, tmp_11, tmp_10)
        tmp_11 = tmp_10 = None
        tmp_67 = torch.nn.functional.gelu(tmp_66)
        tmp_66 = None
        tmp_68 = torch.nn.functional.linear(tmp_67, tmp_17, tmp_16)
        tmp_67 = tmp_17 = tmp_16 = None
        tmp_69 = torch.nn.functional.dropout(tmp_68, 0.0, False, False)
        tmp_68 = None
        tmp_70 = torch.nn.functional.layer_norm(tmp_69, (384,), tmp_13, tmp_12, 1e-05)
        tmp_69 = tmp_13 = tmp_12 = None
        tmp_71 = tmp_65 + tmp_70
        tmp_65 = tmp_70 = None
        tmp_72 = tmp_71.view(1, 64, 64, 384)
        tmp_73 = torch.nn.functional.pad(tmp_72, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_72 = None
        tmp_74 = tmp_73.view(1, 8, 8, 8, 8, 384)
        tmp_73 = None
        tmp_75 = tmp_74.permute(0, 1, 3, 2, 4, 5)
        tmp_74 = None
        tmp_76 = tmp_75.contiguous()
        tmp_75 = None
        tmp_77 = tmp_76.view(-1, 8, 8, 384)
        tmp_76 = None
        tmp_78 = tmp_77.view(-1, 64, 384)
        tmp_77 = None
        return (tmp_78, tmp_71)